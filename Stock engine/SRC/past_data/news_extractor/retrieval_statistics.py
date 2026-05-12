"""
retrieval_statistics.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive retrieval metrics and reporting for the news collection pipeline.

Tracks:
  1. Retrieval funnel (GDELT hits → new URLs → extracted → relevant)
  2. Relevance score distribution (before/after filtering)
  3. Deduplication impact (URLs removed, content dedup rate)
  4. Top rejected articles (for debugging false negatives)
  5. Finance density statistics
  6. Source distribution (trusted vs untrusted domains)
  7. Session lag analysis (pre-market vs intraday vs post-market)

Goal: Understand precision vs recall tradeoffs and calibrate thresholds.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Stats collection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalMetrics:
    """Per-stock retrieval statistics for one collection run."""
    
    stock_symbol: str
    run_date: date
    
    # Funnel stages
    gdelt_hits: int = 0                    # GDELT API results
    gdelt_new_urls: int = 0                # After dedup vs DB
    extract_attempts: int = 0              # URLs sent to extractor
    extract_success: int = 0               # Successful extractions
    relevance_filtered_out: int = 0        # Failed relevance gate
    final_saved: int = 0                   # Saved to DB
    
    # Deduplication breakdown
    dedup_urls_caught: int = 0             # URL hash duplicates
    dedup_content_caught: int = 0          # Content hash duplicates
    dedup_title_caught: int = 0            # Title similarity caught
    
    # Relevance score tracking
    relevance_scores: List[float] = field(default_factory=list)
    finance_densities: List[float] = field(default_factory=list)
    
    # Rejected articles (for debugging)
    rejected_articles: List[Dict] = field(default_factory=list)
    
    # Source distribution
    source_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Session lag distribution
    session_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Credibility distribution
    credibility_scores: List[float] = field(default_factory=list)
    
    # Sponsored vs organic
    sponsored_count: int = 0
    organic_count: int = 0
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # ────────────────────────────────────────────────────────────────────────
    # Derived metrics
    # ────────────────────────────────────────────────────────────────────────
    
    @property
    def gdelt_to_extraction_ratio(self) -> float:
        """How many GDELT hits became extraction attempts (after dedup)."""
        if self.gdelt_hits == 0:
            return 0.0
        return self.extract_attempts / self.gdelt_hits
    
    @property
    def extraction_success_rate(self) -> float:
        """How many extraction attempts succeeded."""
        if self.extract_attempts == 0:
            return 0.0
        return self.extract_success / self.extract_attempts
    
    @property
    def relevance_pass_rate(self) -> float:
        """How many successful extractions passed relevance gate."""
        if self.extract_success == 0:
            return 0.0
        return (self.extract_success - self.relevance_filtered_out) / self.extract_success
    
    @property
    def overall_funnel_rate(self) -> float:
        """GDELT hits → final saved (overall funnel efficiency)."""
        if self.gdelt_hits == 0:
            return 0.0
        return self.final_saved / self.gdelt_hits
    
    @property
    def dedup_catch_rate(self) -> float:
        """How many potential duplicates were caught."""
        if self.gdelt_hits == 0:
            return 0.0
        total_caught = self.dedup_urls_caught + self.dedup_content_caught + self.dedup_title_caught
        return total_caught / self.gdelt_hits
    
    @property
    def avg_relevance_score(self) -> float:
        """Average relevance score across extracted articles."""
        if not self.relevance_scores:
            return 0.0
        return sum(self.relevance_scores) / len(self.relevance_scores)
    
    @property
    def avg_finance_density(self) -> float:
        """Average finance keyword density."""
        if not self.finance_densities:
            return 0.0
        return sum(self.finance_densities) / len(self.finance_densities)
    
    @property
    def avg_credibility(self) -> float:
        """Average source credibility score."""
        if not self.credibility_scores:
            return 0.0
        return sum(self.credibility_scores) / len(self.credibility_scores)
    
    @property
    def elapsed_seconds(self) -> float:
        """Total processing time."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0
    
    @property
    def articles_per_minute(self) -> float:
        """Processing speed."""
        if self.elapsed_seconds == 0:
            return 0.0
        return (self.final_saved / self.elapsed_seconds) * 60
    
    def to_dict(self) -> Dict:
        """Export as flat dictionary."""
        return {
            "stock_symbol": self.stock_symbol,
            "run_date": str(self.run_date),
            "gdelt_hits": self.gdelt_hits,
            "gdelt_new_urls": self.gdelt_new_urls,
            "extract_attempts": self.extract_attempts,
            "extract_success": self.extract_success,
            "extract_success_rate": round(self.extraction_success_rate, 3),
            "relevance_filtered_out": self.relevance_filtered_out,
            "relevance_pass_rate": round(self.relevance_pass_rate, 3),
            "final_saved": self.final_saved,
            "dedup_urls_caught": self.dedup_urls_caught,
            "dedup_content_caught": self.dedup_content_caught,
            "dedup_title_caught": self.dedup_title_caught,
            "overall_funnel_rate": round(self.overall_funnel_rate, 3),
            "avg_relevance_score": round(self.avg_relevance_score, 4),
            "avg_finance_density": round(self.avg_finance_density, 4),
            "avg_credibility": round(self.avg_credibility, 4),
            "sponsored_count": self.sponsored_count,
            "organic_count": self.organic_count,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "articles_per_minute": round(self.articles_per_minute, 2),
            "top_sources": dict(list(sorted(
                self.source_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])),
            "session_distribution": dict(self.session_distribution),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_retrieval_report(metrics: RetrievalMetrics) -> str:
    """
    Generate human-readable retrieval report.
    
    Returns formatted string for logging/display.
    """
    lines = []
    lines.append("=" * 75)
    lines.append(f"RETRIEVAL REPORT | {metrics.stock_symbol} | {metrics.run_date}")
    lines.append("=" * 75)
    
    # Funnel
    lines.append("")
    lines.append("FUNNEL ANALYSIS:")
    lines.append(f"  GDELT hits:               {metrics.gdelt_hits:>6}")
    lines.append(f"    ↓ (dedup)               {metrics.gdelt_new_urls:>6}  "
                 f"(new URLs: {100*metrics.gdelt_new_urls/max(metrics.gdelt_hits,1):.1f}%)")
    lines.append(f"  Extraction attempts:      {metrics.extract_attempts:>6}  "
                 f"({100*metrics.gdelt_to_extraction_ratio:.1f}% of hits)")
    lines.append(f"  Extraction success:       {metrics.extract_success:>6}  "
                 f"({100*metrics.extraction_success_rate:.1f}% success)")
    lines.append(f"  Relevance filtered:       {metrics.relevance_filtered_out:>6}  "
                 f"({100*(1-metrics.relevance_pass_rate):.1f}% rejected)")
    lines.append(f"  ➜ FINAL SAVED:            {metrics.final_saved:>6}  "
                 f"({100*metrics.overall_funnel_rate:.1f}% of GDELT hits)")
    
    # Deduplication
    lines.append("")
    lines.append("DEDUPLICATION BREAKDOWN:")
    lines.append(f"  URL duplicates caught:    {metrics.dedup_urls_caught:>6}")
    lines.append(f"  Content hash duplicates:  {metrics.dedup_content_caught:>6}")
    lines.append(f"  Title similarity caught:  {metrics.dedup_title_caught:>6}")
    total_dup = (metrics.dedup_urls_caught + metrics.dedup_content_caught + 
                 metrics.dedup_title_caught)
    lines.append(f"  Total duplicates caught:  {total_dup:>6}  "
                 f"({100*metrics.dedup_catch_rate:.1f}% of GDELT hits)")
    
    # Quality metrics
    lines.append("")
    lines.append("QUALITY METRICS:")
    lines.append(f"  Avg relevance score:      {metrics.avg_relevance_score:>7.4f}  "
                 f"(range: 0.0-1.0)")
    lines.append(f"  Avg finance density:      {metrics.avg_finance_density:>7.4f}  "
                 f"(fraction of sentences)")
    lines.append(f"  Avg source credibility:   {metrics.avg_credibility:>7.4f}  "
                 f"(Reuters=1.0, unknown=0.3)")
    
    # Article mix
    lines.append("")
    lines.append("ARTICLE MIX:")
    total_articles = metrics.sponsored_count + metrics.organic_count
    if total_articles > 0:
        lines.append(f"  Organic articles:         {metrics.organic_count:>6}  "
                     f"({100*metrics.organic_count/total_articles:.1f}%)")
        lines.append(f"  Sponsored articles:       {metrics.sponsored_count:>6}  "
                     f"({100*metrics.sponsored_count/total_articles:.1f}%)")
    
    # Top sources
    if metrics.source_distribution:
        lines.append("")
        lines.append("TOP 5 SOURCES:")
        for domain, count in list(sorted(
            metrics.source_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        ))[:5]:
            lines.append(f"  {domain:<30} {count:>5}")
    
    # Session distribution
    if metrics.session_distribution:
        lines.append("")
        lines.append("SESSION LAG DISTRIBUTION:")
        for session, count in sorted(metrics.session_distribution.items()):
            lines.append(f"  {session:<20} {count:>6}")
    
    # Performance
    if metrics.elapsed_seconds > 0:
        lines.append("")
        lines.append("PERFORMANCE:")
        lines.append(f"  Elapsed time:             {metrics.elapsed_seconds:>7.1f}s")
        lines.append(f"  Articles/minute:          {metrics.articles_per_minute:>7.1f}")
    
    # Rejected articles (top 3 for debugging)
    if metrics.rejected_articles:
        lines.append("")
        lines.append("TOP REJECTED ARTICLES (low relevance):")
        for i, article in enumerate(metrics.rejected_articles[:3], 1):
            lines.append(f"  {i}. {article.get('title', 'N/A')[:60]}")
            lines.append(f"     Relevance: {article.get('relevance_score', 0):.3f}")
            lines.append(f"     URL: {article.get('url', 'N/A')[:60]}")
    
    lines.append("=" * 75)
    lines.append("")
    
    return "\n".join(lines)


def print_comparison_report(
    before_metrics: List[RetrievalMetrics],
    after_metrics: List[RetrievalMetrics],
) -> str:
    """
    Compare before/after filtering impact.
    
    Shows precision improvements from enhanced filtering.
    """
    lines = []
    lines.append("=" * 75)
    lines.append("BEFORE vs AFTER: HIGH-PRECISION FILTERING IMPACT")
    lines.append("=" * 75)
    
    before_saved = sum(m.final_saved for m in before_metrics)
    after_saved = sum(m.final_saved for m in after_metrics)
    before_avg_rel = (sum(m.avg_relevance_score * m.extract_success 
                          for m in before_metrics) / max(sum(m.extract_success for m in before_metrics), 1))
    after_avg_rel = (sum(m.avg_relevance_score * m.extract_success 
                         for m in after_metrics) / max(sum(m.extract_success for m in after_metrics), 1))
    
    lines.append("")
    lines.append("QUANTITY:")
    lines.append(f"  Before filter:  {before_saved:>6} articles")
    lines.append(f"  After filter:   {after_saved:>6} articles")
    lines.append(f"  Reduction:      {before_saved - after_saved:>6}  "
                 f"({100*(before_saved-after_saved)/max(before_saved,1):.1f}%)")
    
    lines.append("")
    lines.append("QUALITY (avg relevance score):")
    lines.append(f"  Before filter:  {before_avg_rel:>7.4f}")
    lines.append(f"  After filter:   {after_avg_rel:>7.4f}")
    lines.append(f"  Improvement:    {after_avg_rel - before_avg_rel:>7.4f}  "
                 f"({100*(after_avg_rel-before_avg_rel)/max(before_avg_rel,0.001):.1f}%)")
    
    lines.append("")
    lines.append("TRADEOFF:")
    precision_gain = 100 * (after_avg_rel - before_avg_rel) / max(before_avg_rel, 0.001)
    recall_loss = 100 * (before_saved - after_saved) / max(before_saved, 1)
    lines.append(f"  Precision gain:  {precision_gain:>6.1f}%")
    lines.append(f"  Recall loss:     {recall_loss:>6.1f}%")
    lines.append(f"  Ratio:           {precision_gain / max(recall_loss, 0.1):>6.2f}x  "
                 f"(gain per % lost)")
    
    lines.append("=" * 75)
    lines.append("")
    
    return "\n".join(lines)
