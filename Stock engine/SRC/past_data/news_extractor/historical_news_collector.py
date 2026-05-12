"""
historical_news_collector.py
─────────────────────────────────────────────────────────────────────────────
Main orchestrator for the news ingestion pipeline.

Wires together:
  stock_universe  → alias expansion
  gdelt_client    → URL discovery
  article_extractor → text + relevance
  db_schema tables  → persistence

Can be run:
  (a) Historical backfill:  collect 1–3 years of news for all NIFTY50
  (b) Daily incremental:    collect yesterday's news each morning
  (c) Single stock:         useful for debugging / research

Pipeline per stock:
  ─────────────────────────────────────────────────────
  GDELT query (per alias, per date chunk)
       ↓ GDELTArticle list (URL + metadata)
  URL deduplication (MD5 hash vs DB)
       ↓ new URLs only
  Article extraction (trafilatura chain)
       ↓ ExtractionResult objects
  Relevance filter (score ≥ 0.15, words ≥ 80)
       ↓ usable articles
  Sponsored detection (URL + title heuristics)
       ↓
  Market session alignment (IST)
       ↓
  SQLite INSERT → articles table
  ─────────────────────────────────────────────────────

FinBERT is NOT called here — it's a separate step (Sentiment.py).
This module's job is clean article storage, not scoring.

Performance notes:
  ~250 GDELT results per chunk × 15-day chunks × N aliases.
  Extraction success rate typically 55–70% (paywalls, JS sites).
  Realistic throughput: ~80-120 articles/hour per stock.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    from .gdelt_client import GDELTClient, assign_trading_session
    from .article_extractor import ArticleExtractor
    from .stock_universe import (
        NIFTY50, StockProfile, get_stock, get_credibility, is_sponsored
    )
    from .relevance_scorer import FinanceRelevanceScorer, compute_finance_relevance
    from .deduplication import ArticleDeduplicator, url_hash
    from .retrieval_statistics import RetrievalMetrics
except ImportError:
    from gdelt_client import GDELTClient, assign_trading_session
    from article_extractor import ArticleExtractor
    from stock_universe import (
        NIFTY50, StockProfile, get_stock, get_credibility, is_sponsored
    )
    from relevance_scorer import FinanceRelevanceScorer, compute_finance_relevance
    from deduplication import ArticleDeduplicator, url_hash
    from retrieval_statistics import RetrievalMetrics

logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "db" / "stock_engine.db"


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CollectorConfig:
    db_path:             str   = str(DEFAULT_DB_PATH)
    gdelt_chunk_days:    int   = 15
    gdelt_delay_s:       float = 2.0
    extract_delay_s:     float = 0.8     # polite delay between article fetches
    min_relevance:       float = 0.15    # Basic extraction relevance gate
    high_precision_threshold: float = 0.50  # HIGH-PRECISION filter: only save scores ≥ this
    min_word_count:      int   = 80
    max_aliases_per_q:   int   = 4       # top N aliases for GDELT
    dry_run:             bool  = False   # if True: fetch metadata but don't extract/store
    max_articles_per_stock_per_day: int = 50   # hard cap to prevent DB bloat
    auto_init_schema:    bool  = True
    use_finance_scorer:  bool  = True    # Use enhanced finance relevance scorer
    enable_deduplication: bool = True    # Use advanced deduplication


# ─────────────────────────────────────────────────────────────────────────────
# Collector stats (for logging + run_log table)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunStats:
    stock_symbol:      str
    date_range_start:  date
    date_range_end:    date
    gdelt_hits:        int = 0
    gdelt_new_urls:    int = 0
    extract_attempts:  int = 0
    extract_success:   int = 0
    relevance_passed:  int = 0
    articles_saved:    int = 0
    sponsored_count:   int = 0
    organic_count:     int = 0
    errors:            int = 0
    # Enhanced metrics
    high_precision_filtered: int = 0     # Articles rejected by high-precision gate
    dedup_urls_caught: int = 0
    dedup_content_caught: int = 0
    dedup_title_caught: int = 0
    relevance_scores: List[float] = field(default_factory=list)
    finance_densities: List[float] = field(default_factory=list)
    credibility_scores: List[float] = field(default_factory=list)
    rejected_articles: List[Dict] = field(default_factory=list)
    started_at:        Optional[datetime] = None
    completed_at:      Optional[datetime] = None

    @property
    def extract_success_rate(self) -> float:
        if self.extract_attempts == 0:
            return 0.0
        return round(self.extract_success / self.extract_attempts, 3)

    @property
    def elapsed_s(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

    def summary(self) -> str:
        return (
            f"[{self.stock_symbol}] "
            f"gdelt={self.gdelt_hits} → new={self.gdelt_new_urls} → "
            f"extract={self.extract_success}/{self.extract_attempts} → "
            f"relevance_passed={self.relevance_passed} → "
            f"high_precision={self.articles_saved} "
            f"(organic={self.organic_count}, sponsored={self.sponsored_count}) "
            f"in {self.elapsed_s:.0f}s"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main Collector
# ─────────────────────────────────────────────────────────────────────────────

class NewsCollector:
    """
    Main ingestion pipeline.

    Usage — historical backfill:
        collector = NewsCollector(config)
        collector.backfill(
            symbols    = ["TCS", "INFY", "RELIANCE"],
            start_date = date(2022, 1, 1),
            end_date   = date(2024, 4, 15),
        )

    Usage — daily incremental (call from cron/app.py):
        collector = NewsCollector(config)
        collector.collect_daily(target_date=date.today())
    """

    def __init__(self, config: Optional[CollectorConfig] = None):
        self.config = config or CollectorConfig()
        self.config.db_path = str(Path(self.config.db_path).expanduser().resolve())
        if self.config.auto_init_schema:
            self._ensure_schema()
        self.gdelt     = GDELTClient(
            request_delay = self.config.gdelt_delay_s,
            chunk_days    = self.config.gdelt_chunk_days,
        )
        self.extractor = ArticleExtractor(
            min_word_count = self.config.min_word_count,
        )
        # Initialize enhanced modules
        self.finance_scorer = (
            FinanceRelevanceScorer(high_precision_threshold=self.config.high_precision_threshold)
            if self.config.use_finance_scorer
            else None
        )
        self.deduplicator = (
            ArticleDeduplicator()
            if self.config.enable_deduplication
            else None
        )
        self._existing_urls: Set[str] = set()

    def _ensure_schema(self):
        """Create base tables + migration columns required by this collector."""
        db_dir = Path(self.config.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        try:
            from SRC.common.db_schema import DatabaseSchema
        except ImportError:
            import sys

            src_root = Path(__file__).resolve().parents[2]
            if str(src_root) not in sys.path:
                sys.path.insert(0, str(src_root))
            try:
                from SRC.common.db_schema import DatabaseSchema
            except ImportError:
                logger.warning(
                    "common.db_schema not importable; skipping auto schema initialization."
                )
                return

        try:
            DatabaseSchema(db_path=self.config.db_path).init_schema(verbose=False)
        except Exception as exc:
            logger.warning(f"Schema init skipped due to error: {exc}")

        try:
            try:
                from .migrate_articles_schema import migrate
            except ImportError:
                from migrate_articles_schema import migrate
            migrate(self.config.db_path, verbose=False)
        except Exception as exc:
            logger.warning(f"Articles migration skipped due to error: {exc}")

    # ── Public API ─────────────────────────────────────────────────────────

    def backfill(
        self,
        symbols:    Optional[List[str]] = None,
        start_date: Optional[date]      = None,
        end_date:   Optional[date]      = None,
    ) -> List[RunStats]:
        """
        Run historical backfill for given symbols and date range.
        Processes stocks sequentially (GDELT is free — don't parallelize).
        """
        symbols    = symbols or [s.symbol for s in NIFTY50]
        end_date   = end_date   or date.today()
        start_date = start_date or (end_date - timedelta(days=365))

        logger.info(
            f"[BACKFILL] {len(symbols)} stocks | "
            f"{start_date} → {end_date} | db={self.config.db_path}"
        )

        all_stats = []
        for i, sym in enumerate(symbols):
            stock = get_stock(sym)
            if not stock:
                logger.warning(f"Unknown symbol: {sym}")
                continue

            logger.info(f"[{i+1}/{len(symbols)}] Processing {sym}")
            stats = self._collect_for_stock(
                stock      = stock,
                start_date = datetime.combine(start_date, datetime.min.time()),
                end_date   = datetime.combine(end_date,   datetime.max.time()),
            )
            all_stats.append(stats)
            self._write_run_log(stats)

        self._print_summary(all_stats)
        return all_stats

    def collect_daily(
        self,
        target_date: Optional[date] = None,
        symbols:     Optional[List[str]] = None,
    ) -> List[RunStats]:
        """
        Collect news for a single day (default: yesterday).
        Designed to be called from app.py at market open (~09:00 IST).
        """
        target_date = target_date or (date.today() - timedelta(days=1))
        start_dt    = datetime.combine(target_date, datetime.min.time())
        end_dt      = datetime.combine(target_date, datetime.max.time())
        symbols     = symbols or [s.symbol for s in NIFTY50]

        logger.info(f"[DAILY] {target_date} | {len(symbols)} stocks")
        all_stats = []
        for sym in symbols:
            stock = get_stock(sym)
            if not stock:
                continue
            stats = self._collect_for_stock(stock, start_dt, end_dt)
            all_stats.append(stats)
            self._write_run_log(stats)

        return all_stats

    # ── Core pipeline ──────────────────────────────────────────────────────

    def _collect_for_stock(
        self,
        stock:      StockProfile,
        start_date: datetime,
        end_date:   datetime,
    ) -> RunStats:
        stats = RunStats(
            stock_symbol     = stock.symbol,
            date_range_start = start_date.date(),
            date_range_end   = end_date.date(),
            started_at       = datetime.utcnow(),
        )

        # Pre-load known URLs to avoid re-fetching
        self._existing_urls = self._load_existing_urls(stock.symbol)

        # ── Step 1: GDELT discovery ──────────────────────────────────────
        query_terms = stock.gdelt_query_terms(max_terms=self.config.max_aliases_per_q)
        gdelt_articles = self.gdelt.fetch_for_stock(
            stock_symbol = stock.symbol,
            query_terms  = query_terms,
            start_date   = start_date,
            end_date     = end_date,
        )
        stats.gdelt_hits = len(gdelt_articles)

        # ── Step 2: Filter to new URLs ────────────────────────────────────
        new_articles = [
            a for a in gdelt_articles
            if a.url_hash not in self._existing_urls and a.language == "English"
        ]
        stats.gdelt_new_urls = len(new_articles)
        logger.info(
            f"[{stock.symbol}] GDELT: {stats.gdelt_hits} total, "
            f"{stats.gdelt_new_urls} new"
        )

        if not new_articles or self.config.dry_run:
            stats.completed_at = datetime.utcnow()
            return stats

        # ── Step 3: Extract article text + Enhanced filtering pipeline ────────
        records_to_save = []

        for gdelt_art in new_articles:
            stats.extract_attempts += 1

            # Advanced deduplication (before extraction to save CPU)
            if self.deduplicator:
                if self.deduplicator.is_duplicate(
                    url=gdelt_art.url,
                    title=gdelt_art.title,
                ):
                    stats.dedup_urls_caught += 1
                    logger.debug(f"[{stock.symbol}] Dedup caught: {gdelt_art.url[:60]}")
                    time.sleep(self.config.extract_delay_s)
                    continue

            result = self.extractor.extract(
                url          = gdelt_art.url,
                stock_symbol = stock.symbol,
                aliases      = stock.aliases,
            )

            if not result.extraction_ok:
                stats.errors += 1
                time.sleep(self.config.extract_delay_s)
                continue

            stats.extract_success += 1

            # ── Step 4: Basic relevance gate (extraction scorer) ────────────────
            if result.relevance_score < self.config.min_relevance:
                logger.debug(
                    f"[{stock.symbol}] Low basic relevance ({result.relevance_score:.2f}): "
                    f"{gdelt_art.url[:70]}"
                )
                time.sleep(self.config.extract_delay_s)
                continue

            stats.relevance_passed += 1
            stats.relevance_scores.append(result.relevance_score)
            stats.finance_densities.append(result.finance_density)

            # ── Step 5: ENHANCED relevance scoring (high-precision gate) ────────
            enhanced_relevance = result.relevance_score
            if self.finance_scorer:
                finance_breakdown = self.finance_scorer.score(
                    title=result.title or gdelt_art.title,
                    text=result.text,
                    url=gdelt_art.url,
                    stock_symbol=stock.symbol,
                    aliases=stock.aliases,
                    query_alias_used=gdelt_art.query_term,
                )
                enhanced_relevance = finance_breakdown.total_score
                logger.debug(
                    f"[{stock.symbol}] Enhanced relevance: "
                    f"basic={result.relevance_score:.3f} → "
                    f"finance={enhanced_relevance:.3f} "
                    f"(earnings={finance_breakdown.earnings_signal:.2f}, "
                    f"source={finance_breakdown.source_credibility:.2f})"
                )

            # HIGH-PRECISION filter: only save articles with score >= threshold
            if enhanced_relevance < self.config.high_precision_threshold:
                stats.high_precision_filtered += 1
                logger.debug(
                    f"[{stock.symbol}] Filtered by high-precision gate: "
                    f"{enhanced_relevance:.3f} < {self.config.high_precision_threshold}"
                )
                # Track rejected article for debugging
                if len(stats.rejected_articles) < 10:
                    stats.rejected_articles.append({
                        "title": result.title or gdelt_art.title,
                        "url": gdelt_art.url,
                        "relevance_score": enhanced_relevance,
                        "reason": "high_precision_threshold",
                    })
                time.sleep(self.config.extract_delay_s)
                continue

            # ── Step 6: Enrich metadata ───────────────────────────────────────
            pub_utc = gdelt_art.published_dt()
            session_info = (
                assign_trading_session(pub_utc)
                if pub_utc
                else {"trading_date": None, "session_lag": "UNKNOWN", "published_ist": None}
            )

            spons = is_sponsored(gdelt_art.url, result.title)
            cred  = get_credibility(gdelt_art.url, stock)
            stats.credibility_scores.append(cred)

            record = {
                # articles table columns
                "collection_date":      session_info.get("trading_date"),
                "stock_symbol":         stock.symbol,
                "title":                result.title or gdelt_art.title,
                "headline_url":         gdelt_art.url,
                "article_text":         result.text,
                "published_time":       session_info.get("published_ist"),
                "fetched_time":         datetime.utcnow().isoformat(),
                "is_sponsored":         spons,
                "source":               gdelt_art.domain,
                "source_credibility":   cred,
                "session_lag":          session_info.get("session_lag"),
                "relevance_score":      enhanced_relevance,  # Use enhanced score in DB
                "finance_density":      result.finance_density,
                "word_count":           result.word_count,
                "gdelt_tone":           gdelt_art.gdelt_tone,
                # FinBERT fields — populated later by Sentiment.py
                "sentiment_score":              None,
                "sentiment_positive_prob":      None,
                "sentiment_negative_prob":      None,
                "sentiment_neutral_prob":       None,
            }

            records_to_save.append(record)

            if spons:
                stats.sponsored_count += 1
            else:
                stats.organic_count += 1

            # Mark as seen in deduplicator and DB tracker
            self._existing_urls.add(gdelt_art.url_hash)
            if self.deduplicator:
                self.deduplicator.mark_seen(
                    url=gdelt_art.url,
                    title=result.title or gdelt_art.title,
                    text=result.text,
                )

            time.sleep(self.config.extract_delay_s)

            # Cap per-stock per-run
            if len(records_to_save) >= self.config.max_articles_per_stock_per_day:
                logger.info(f"[{stock.symbol}] Hit daily cap; stopping extraction")
                break

        # ── Step 7: Batch insert to DB ─────────────────────────────────────
        if records_to_save:
            saved = self._save_articles(records_to_save)
            stats.articles_saved = saved

        stats.completed_at = datetime.utcnow()
        logger.info(stats.summary())
        return stats

    # ── Database helpers ───────────────────────────────────────────────────

    @contextmanager
    def _db(self):
        Path(self.config.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.config.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _load_existing_urls(self, stock_symbol: str) -> Set[str]:
        """Load MD5 hashes of all known URLs for this stock (avoids re-fetch)."""
        import hashlib
        try:
            with self._db() as conn:
                rows = conn.execute(
                    "SELECT headline_url FROM articles WHERE stock_symbol = ?",
                    (stock_symbol,)
                ).fetchall()
            return {
                hashlib.md5(r["headline_url"].encode("utf-8", errors="ignore")).hexdigest()
                for r in rows
                if r["headline_url"]
            }
        except Exception:
            return set()

    def _save_articles(self, records: List[Dict[str, Any]]) -> int:
        """Bulk insert articles, skipping duplicates (INSERT OR IGNORE)."""
        if not records:
            return 0

        # Build dynamic column list from first record
        cols   = list(records[0].keys())
        placeholders = ", ".join("?" * len(cols))
        col_str      = ", ".join(cols)
        sql = f"INSERT OR IGNORE INTO articles ({col_str}) VALUES ({placeholders})"

        rows = [tuple(r[c] for c in cols) for r in records]

        try:
            with self._db() as conn:
                before = conn.total_changes
                conn.executemany(sql, rows)
                inserted = conn.total_changes - before
            return inserted
        except Exception as e:
            logger.error(f"DB insert failed: {e}")
            return 0

    def _write_run_log(self, stats: RunStats):
        """Write a summary row to run_log table."""
        try:
            with self._db() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO run_log
                    (run_date, run_type, status, stocks_processed,
                     articles_collected, started_at, completed_at, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(stats.date_range_end),
                    "news_collection",
                    "success" if stats.errors == 0 else "partial",
                    1,
                    stats.articles_saved,
                    stats.started_at.isoformat() if stats.started_at else None,
                    stats.completed_at.isoformat() if stats.completed_at else None,
                    None,
                ))
        except Exception as e:
            logger.error(f"run_log write failed: {e}")

    @staticmethod
    def _print_summary(all_stats: List[RunStats]):
        total_gdelt    = sum(s.gdelt_hits for s in all_stats)
        total_extracted = sum(s.extract_success for s in all_stats)
        total_relevance_passed = sum(s.relevance_passed for s in all_stats)
        total_high_precision_filtered = sum(s.high_precision_filtered for s in all_stats)
        total_saved    = sum(s.articles_saved for s in all_stats)
        total_organic  = sum(s.organic_count for s in all_stats)
        total_sponsored = sum(s.sponsored_count for s in all_stats)
        
        # Compute averages
        avg_extract_ok = (
            sum(s.extract_success_rate for s in all_stats) / len(all_stats)
            if all_stats else 0
        )
        all_relevance_scores = []
        all_finance_densities = []
        for s in all_stats:
            all_relevance_scores.extend(s.relevance_scores)
            all_finance_densities.extend(s.finance_densities)
        
        avg_relevance = (
            sum(all_relevance_scores) / len(all_relevance_scores)
            if all_relevance_scores else 0
        )
        avg_finance_density = (
            sum(all_finance_densities) / len(all_finance_densities)
            if all_finance_densities else 0
        )
        
        print("\n" + "="*75)
        print("NEWS COLLECTION SUMMARY — HIGH-PRECISION FILTERING")
        print("="*75)
        print(f"  Stocks processed:          {len(all_stats)}")
        
        print("\n  RETRIEVAL FUNNEL:")
        print(f"    GDELT hits:            {total_gdelt:>6}")
        print(f"    ↓ Extracted:           {total_extracted:>6}  ({100*total_extracted/max(total_gdelt,1):.1f}%)")
        print(f"    ↓ Basic relevance:     {total_relevance_passed:>6}  ({100*total_relevance_passed/max(total_extracted,1):.1f}%)")
        print(f"    ↓ High-precision gate: {total_saved:>6}  ({100*total_saved/max(total_relevance_passed,1):.1f}%)")
        print(f"    ➜ FINAL SAVED:         {total_saved:>6}  ({100*total_saved/max(total_gdelt,1):.1f}% of GDELT hits)")
        
        print("\n  FILTERING IMPACT:")
        print(f"    Filtered by high-prec: {total_high_precision_filtered:>6}")
        print(f"    Precision trade-off:   {100*total_high_precision_filtered/max(total_relevance_passed,1):.1f}% lost for quality")
        
        print("\n  QUALITY METRICS:")
        print(f"    Avg relevance score:   {avg_relevance:>7.4f}")
        print(f"    Avg finance density:   {avg_finance_density:>7.4f}")
        print(f"    Avg extraction rate:   {avg_extract_ok:>7.1%}")
        
        print("\n  ARTICLE MIX:")
        total_articles = total_organic + total_sponsored
        if total_articles > 0:
            print(f"    Organic:               {total_organic:>6}  ({100*total_organic/total_articles:.1f}%)")
            print(f"    Sponsored:             {total_sponsored:>6}  ({100*total_sponsored/total_articles:.1f}%)")
            if total_sponsored > 0:
                ratio = total_organic / total_sponsored
                print(f"    Ratio (organic/sponsored): {ratio:>6.2f}x")
        
        print("="*75 + "\n")


HistoricalNewsCollector = NewsCollector


# ─────────────────────────────────────────────────────────────────────────────
# CLI interface
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="News Collection Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # ── backfill ─────────────────────────────────────
    bp = subparsers.add_parser("backfill", help="Historical news backfill")
    bp.add_argument("--symbols", nargs="+", help="NSE symbols (default: all NIFTY50)")
    bp.add_argument("--start",   required=True, help="Start date YYYY-MM-DD")
    bp.add_argument("--end",     help="End date YYYY-MM-DD (default: today)")
    bp.add_argument("--db",      default=str(DEFAULT_DB_PATH))
    bp.add_argument("--dry-run", action="store_true")

    # ── daily ─────────────────────────────────────────
    dp = subparsers.add_parser("daily", help="Yesterday's news")
    dp.add_argument("--symbols", nargs="+")
    dp.add_argument("--date",    help="YYYY-MM-DD (default: yesterday)")
    dp.add_argument("--db",      default=str(DEFAULT_DB_PATH))

    args = parser.parse_args()

    if args.command == "backfill":
        config = CollectorConfig(
            db_path  = args.db,
            dry_run  = args.dry_run,
        )
        collector = NewsCollector(config)
        collector.backfill(
            symbols    = args.symbols,
            start_date = date.fromisoformat(args.start),
            end_date   = date.fromisoformat(args.end) if args.end else date.today(),
        )

    elif args.command == "daily":
        config = CollectorConfig(db_path=args.db)
        collector = NewsCollector(config)
        target = date.fromisoformat(args.date) if args.date else date.today() - timedelta(days=1)
        collector.collect_daily(target_date=target, symbols=args.symbols)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
