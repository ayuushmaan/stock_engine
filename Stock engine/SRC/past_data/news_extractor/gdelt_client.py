"""
gdelt_client.py
─────────────────────────────────────────────────────────────────────────────
GDELT 2.0 Document API client.

Design decisions (quant rationale):
  1. Query per alias, not per stock — maximises recall.
  2. 15-day date chunks — GDELT's ArtList caps at 250 results/query;
     chunking ensures we don't miss bursts (earnings, scandals).
  3. Dedup within client — same URL can appear across multiple alias queries.
  4. Rate limit: 1 req / 2s — GDELT is free; be a good citizen.
  5. Store raw GDELT metadata (tone, themes) — useful as free features
     alongside FinBERT sentiment.

GDELT tone field: composite score ∈ [-100, +100].
  Can be used as a WEAK signal sanity check against FinBERT output.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

GDELT_DOC_API   = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_DATE_FMT  = "%Y%m%d%H%M%S"       # GDELT's required datetime format
CHUNK_DAYS      = 15                    # Days per query window
MAX_RECORDS     = 250                   # GDELT ArtList hard cap
REQUEST_DELAY   = 5.0                   # Seconds between requests
REQUEST_TIMEOUT = 20                    # HTTP timeout

# Finance keyword constraints for high-precision retrieval
# Used to filter out irrelevant global "Reliance" mentions (movies, politics, agriculture)
FINANCE_KEYWORDS = [
    "stock", "shares", "market", "investor", "revenue",
    "earnings", "quarter", "profit", "margin", "EBITDA",
    "NSE", "BSE", "listing", "IPO", "FII", "DII",
    "price", "valuation", "acquisition", "merger",
    "guidance", "outlook", "target", "upgrade", "downgrade",
    "trading", "dividend", "buyback", "capital"
]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

def generate_finance_constrained_query(
    query_term: str,
    finance_keywords: List[str] = FINANCE_KEYWORDS,
    language_filter: str = "English",
) -> str:
    """
    Generate a GDELT-compatible boolean query with finance constraints.
    
    Reduces false positives by requiring the stock mention to be in a
    finance-relevant context (e.g., prevents "Reliance" movie mentions).
    
    Pattern: ("Company Name") AND (finance_keyword1 OR finance_keyword2 OR ...)
    
    Args:
        query_term: Stock name/alias (e.g. "Reliance Industries")
        finance_keywords: List of finance domain keywords to OR together
        language_filter: Language filter for GDELT (e.g. "English")
    
    Returns:
        Boolean query string safe for GDELT API
    """
    # Escape quotes and build phrase query
    safe_term = query_term.replace('"', '\\"')
    
    # Build finance keyword OR clause
    kw_clause = " OR ".join(finance_keywords[:12])  # Limit to 12 to keep query size reasonable
    
    # Combine with AND operator
    query = f'("{safe_term}") AND ({kw_clause}) sourcelang:{language_filter}'
    
    return query


class GDELTArticle:
    """Lightweight container for one GDELT result row."""

    __slots__ = [
        "url", "title", "seendate", "domain", "language",
        "sourcecountry", "gdelt_tone", "gdelt_themes",
        "query_term", "stock_symbol", "url_hash"
    ]

    def __init__(self, raw: Dict[str, Any], query_term: str, stock_symbol: str):
        self.url           = raw.get("url", "")
        self.title         = raw.get("title", "")
        self.seendate      = raw.get("seendate", "")
        self.domain        = raw.get("domain", "")
        self.language      = raw.get("language", "")
        self.sourcecountry = raw.get("sourcecountry", "")
        self.gdelt_tone    = self._parse_tone(raw.get("tone"))
        self.gdelt_themes  = raw.get("themes", "")
        self.query_term    = query_term
        self.stock_symbol  = stock_symbol
        self.url_hash      = hashlib.md5(self.url.encode()).hexdigest()

    @staticmethod
    def _parse_tone(val: Any) -> Optional[float]:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def published_dt(self) -> Optional[datetime]:
        try:
            return datetime.strptime(self.seendate, "%Y%m%dT%H%M%SZ")
        except ValueError:
            return None

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}

    def __repr__(self) -> str:
        return (
            f"<GDELTArticle "
            f"{self.stock_symbol} | "
            f"{self.seendate} | "
            f"{self.title[:60]}>"
        )
    
    
    __slots__ = [
        "url", "title", "seendate", "domain", "language",
        "sourcecountry", "gdelt_tone", "gdelt_themes",
        "query_term", "stock_symbol", "url_hash"
    ]

    def __init__(self, raw: Dict[str, Any], query_term: str, stock_symbol: str):
        self.url           = raw.get("url", "")
        self.title         = raw.get("title", "")
        self.seendate      = raw.get("seendate", "")          # "YYYYMMDDTHHMMSSZ"
        self.domain        = raw.get("domain", "")
        self.language      = raw.get("language", "")
        self.sourcecountry = raw.get("sourcecountry", "")
        self.gdelt_tone    = self._parse_tone(raw.get("tone"))
        self.gdelt_themes  = raw.get("themes", "")
        self.query_term    = query_term
        self.stock_symbol  = stock_symbol
        self.url_hash      = hashlib.md5(self.url.encode()).hexdigest()

    @staticmethod
    def _parse_tone(val: Any) -> Optional[float]:
        """GDELT tone is sometimes packed into unexpected fields; return None if absent."""
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def published_dt(self) -> Optional[datetime]:
        """Parse GDELT seendate → Python datetime (UTC)."""
        try:
            # Format: "20240415T120000Z"
            return datetime.strptime(self.seendate, "%Y%m%dT%H%M%SZ")
        except ValueError:
            return None

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}

    def __repr__(self) -> str:
        return f"<GDELTArticle {self.stock_symbol} | {self.seendate} | {self.title[:60]}>"


# ─────────────────────────────────────────────────────────────────────────────
# GDELT Client
# ─────────────────────────────────────────────────────────────────────────────

class GDELTClient:
    """
    Wrapper around GDELT 2.0 Document API.

    Usage:
        client = GDELTClient()
        articles = client.fetch_for_stock(
            stock_symbol  = "TCS",
            query_terms   = ["Tata Consultancy Services", "TCS", "Tata IT"],
            start_date    = date(2023, 1, 1),
            end_date      = date(2024, 4, 15),
        )
    """

    def __init__(
        self,
        request_delay: float = REQUEST_DELAY,
        chunk_days:    int   = CHUNK_DAYS,
        max_records:   int   = MAX_RECORDS,
        language_filter: str = "English",
        finance_keywords: List[str] = None,
    ):
        self.delay          = request_delay
        self.chunk_days     = chunk_days
        self.max_records    = max_records
        self.language_filter = language_filter
        self.finance_keywords = finance_keywords or FINANCE_KEYWORDS
        self._session       = requests.Session()
        self._session.headers.update({"User-Agent": "StockSentimentResearch/3.0"})
        self._last_request  = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_for_stock(
        self,
        stock_symbol: str,
        query_terms:  List[str],
        start_date:   datetime,
        end_date:     datetime,
    ) -> List[GDELTArticle]:
        """
        Fetch all GDELT articles for a stock across the full date range.
        Runs one query per alias × date chunk.  Deduplicates by URL.

        Returns list of GDELTArticle objects, sorted by seendate desc.
        """
        seen_hashes: set = set()
        results: List[GDELTArticle] = []

        for term in query_terms:
            time.sleep(3)
            logger.info(f"[GDELT] {stock_symbol} | term='{term}'")
            for chunk_start, chunk_end in self._date_chunks(start_date, end_date):
                batch = self._query_single(
                    stock_symbol = stock_symbol,
                    query_term   = term,
                    start_dt     = chunk_start,
                    end_dt       = chunk_end,
                )
                for art in batch:
                    if art.url_hash not in seen_hashes:
                        seen_hashes.add(art.url_hash)
                        results.append(art)

        results.sort(key=lambda a: a.seendate, reverse=True)
        logger.info(f"[GDELT] {stock_symbol} → {len(results)} unique articles "
                    f"({start_date.date()} → {end_date.date()})")
        return results

    def fetch_date_range_generator(
        self,
        stock_symbol: str,
        query_terms:  List[str],
        start_date:   datetime,
        end_date:     datetime,
    ) -> Generator[List[GDELTArticle], None, None]:
        """
        Generator variant — yields one chunk at a time.
        Memory-efficient for long date ranges (e.g. 3-year backtests).
        """
        seen_hashes: set = set()
        for term in query_terms:
            time.sleep(3)
            for chunk_start, chunk_end in self._date_chunks(start_date, end_date):
                batch = self._query_single(stock_symbol, term, chunk_start, chunk_end)
                fresh = [a for a in batch if a.url_hash not in seen_hashes]
                for art in fresh:
                    seen_hashes.add(art.url_hash)
                if fresh:
                    yield fresh

    # ── Internal ─────────────────────────────────────────────────────────────
    def _query_single(
        self,
        stock_symbol: str,
        query_term: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> List[GDELTArticle]:
        """One GDELT API call; returns list of GDELTArticle objects."""

        self._throttle()

        # Generate finance-constrained query to improve precision
        finance_query = generate_finance_constrained_query(
            query_term=query_term,
            finance_keywords=self.finance_keywords,
            language_filter=self.language_filter,
        )

        params = {
            "query": finance_query,
            "mode": "ArtList",
            "maxrecords": self.max_records,
            "format": "json",
            "sort": "DateDesc",
            "startdatetime": start_dt.strftime(GDELT_DATE_FMT),
            "enddatetime": end_dt.strftime(GDELT_DATE_FMT),
        }

        max_retries = 4

        for attempt in range(max_retries):

            try:

                logger.info(
                    f"[GDELT QUERY] "
                    f"{query_term} | "
                    f"{start_dt.date()} → {end_dt.date()} | "
                    f"attempt={attempt + 1}"
                )

                resp = self._session.get(
                    GDELT_DOC_API,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )

                # Handle rate limiting
                if resp.status_code == 429:

                    wait = (2 ** attempt) + 2

                    logger.warning(
                        f"[GDELT] 429 rate limit | "
                        f"retry {attempt + 1}/{max_retries} | "
                        f"sleep={wait}s"
                    )

                    time.sleep(wait)
                    continue

                resp.raise_for_status()

                # Empty response protection
                if not resp.text.strip():

                    logger.warning("[GDELT] Empty response")
                    return []

                # Validate content type
                content_type = resp.headers.get("Content-Type", "")

                if "application/json" not in content_type:

                    logger.warning(
                        f"[GDELT] Non-JSON response: {content_type}"
                    )

                    return []

                data = resp.json()

                articles = data.get("articles", [])

                if not articles:
                    return []

                return [
                    GDELTArticle(
                        raw=a,
                        query_term=query_term,
                        stock_symbol=stock_symbol,
                    )
                    for a in articles
                    if a.get("url")
                ]

            except requests.exceptions.Timeout:

                logger.warning(
                    f"[GDELT] Timeout | "
                    f"{query_term} | "
                    f"attempt={attempt + 1}"
                )

                time.sleep(2 ** attempt)

            except requests.exceptions.HTTPError as e:

                status = (
                    e.response.status_code
                    if e.response is not None
                    else "UNKNOWN"
                )

                logger.warning(
                    f"[GDELT] HTTP {status} | "
                    f"{query_term}"
                )

                time.sleep(2 ** attempt)

            except Exception as e:

                logger.error(
                    f"[GDELT] Unexpected error | "
                    f"{query_term} | "
                    f"{e}"
                )

                time.sleep(2 ** attempt)

        return []
    def _throttle(self):
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.time()

    def _date_chunks(
        self,
        start: datetime,
        end:   datetime,
    ) -> Generator[tuple[datetime, datetime], None, None]:
        """
        Yields (chunk_start, chunk_end) pairs of size self.chunk_days.
        Last chunk is truncated to end.
        """
        cursor = start
        delta  = timedelta(days=self.chunk_days)
        while cursor < end:
            chunk_end = min(cursor + delta, end)
            # GDELT needs end > start
            if chunk_end > cursor:
                yield cursor, chunk_end
            cursor = chunk_end

    def get_rate_stats(self) -> Dict[str, Any]:
        return {
            "delay_s":    self.delay,
            "chunk_days": self.chunk_days,
            "max_records": self.max_records,
            "approx_rpm": round(60 / self.delay, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Market session alignment (IST)
# ─────────────────────────────────────────────────────────────────────────────

IST_OPEN  = {"hour": 9,  "minute": 15}   # 09:15 IST
IST_CLOSE = {"hour": 15, "minute": 30}   # 15:30 IST
IST_OFFSET = timedelta(hours=5, minutes=30)


def assign_trading_session(published_utc: datetime) -> Dict[str, Any]:
    """
    Given a UTC publication timestamp, returns:
      - trading_date: the NSE session date this article is associated with
      - session_lag:  "PRE_MARKET" | "INTRADAY" | "POST_MARKET"

    Logic:
      Article published between 15:30 and 09:15 next day (IST)
          → associated with NEXT trading day
      Article published 09:15–15:30 IST
          → associated with SAME day (intraday)

    This matters for calibration: post-market news has overnight to act.
    Pre-market news has limited time before open.
    """
    pub_ist = published_utc + IST_OFFSET
    market_open  = pub_ist.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = pub_ist.replace(hour=15, minute=30, second=0, microsecond=0)

    if pub_ist < market_open:
        # Pre-market: article affects same day's open
        session_lag  = "PRE_MARKET"
        trading_date = pub_ist.date()
    elif pub_ist <= market_close:
        # Intraday
        session_lag  = "INTRADAY"
        trading_date = pub_ist.date()
    else:
        # Post-market: next trading day
        session_lag  = "POST_MARKET"
        trading_date = (pub_ist + timedelta(days=1)).date()

    return {
        "trading_date":     trading_date,
        "session_lag":      session_lag,
        "published_ist":    pub_ist.isoformat(),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = GDELTClient()
    print("Rate stats:", client.get_rate_stats())

    # Quick smoke test
    from datetime import date
    results = client.fetch_for_stock(
        stock_symbol = "TCS",
        query_terms  = ["Tata Consultancy Services"],
        start_date   = datetime(2024, 4, 1),
        end_date     = datetime(2024, 4, 7),
    )
    print(f"Fetched {len(results)} articles")
    for r in results[:3]:
        print(f"  {r}")
