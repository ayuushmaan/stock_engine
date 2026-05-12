"""
relevance_scorer.py
─────────────────────────────────────────────────────────────────────────────
Enhanced heuristic relevance scoring for high-precision article filtering.

Scoring signals:
  1. Exact company name in title           → +0.25
  2. Exact company name frequency in body  → +0.20
  3. Finance keyword density               → +0.15
  4. NSE/BSE mention                       → +0.10
  5. Earnings/profit/revenue mentions      → +0.10
  6. Trusted finance source weighting      → +0.10
  7. Company name in first paragraph       → +0.05
  8. URL relevance (domain + keywords)     → +0.05

Final score ∈ [0, 1] (normalised).

Rationale:
  The existing relevance_score in article_extractor.py (~0.4 weight for title)
  is good but biased toward title mentions. This module adds:
  - Source credibility weighting (Reuters > local blog)
  - Finance keyword context (prevents false positives)
  - Earnings/profit signals (highest predictive power for stock movement)
  - URL domain pattern matching (economictimes.com > random blog)

Do NOT replace article_extractor's extraction — use this as a POST-EXTRACTION
filter to boost precision before FinBERT scoring.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Signal keywords
# ─────────────────────────────────────────────────────────────────────────────

# Core finance signals (highest relevance when present)
EARNINGS_KEYWORDS = {
    "earnings", "earnings report", "q1", "q2", "q3", "q4",
    "quarterly results", "full year results", "fy", "profit",
    "revenue", "income", "guidance", "forecast", "outlook",
    "beat", "miss", "surprise", "estimate", "consensus",
}

# Stock market context signals
MARKET_KEYWORDS = {
    "stock", "shares", "share price", "trading",
    "nse", "bse", "listing", "ipo", "fii", "dii",
    "market cap", "valuation", "investor", "shareholders",
}

# Financial metrics signals
METRICS_KEYWORDS = {
    "ebitda", "margin", "profit margin", "return", "roe", "roa",
    "debt", "capex", "operating profit", "net profit", "order book",
    "target price", "price target", "buy", "sell", "hold", "upgrade", "downgrade"
}

# Trusted finance news domains (weights for source credibility)
TRUSTED_FINANCE_SOURCES = {
    "reuters.com":            1.0,
    "bloomberg.com":          1.0,
    "ft.com":                 1.0,
    "wsj.com":                1.0,
    "economictimes.com":      0.90,
    "financialexpress.com":   0.85,
    "moneycontrol.com":       0.85,
    "livemint.com":           0.85,
    "bseindia.com":           0.85,
    "nseindia.com":           0.85,
    "cnbctv18.com":           0.80,
    "thehindu.com":           0.75,
    "business-standard.com":  0.75,
}


# ─────────────────────────────────────────────────────────────────────────────
# Result class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RelevanceScoreBreakdown:
    """Detailed scoring breakdown for debugging/analysis."""
    
    total_score: float                     # [0, 1] final score
    title_mention: float                   # Is exact company name in title?
    frequency_signal: float                # Mention frequency in body
    finance_density_signal: float          # Finance keyword density
    market_mention_signal: float           # Stock market context present?
    earnings_signal: float                 # Earnings/profit mentions?
    source_credibility: float              # Trust weight of domain
    first_para_signal: float               # Company name in first 200 words?
    url_relevance_signal: float            # URL domain/path keywords?
    
    query_alias_used: Optional[str] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Export as flat dict for logging."""
        return {
            "total_score": round(self.total_score, 4),
            "title_mention": round(self.title_mention, 4),
            "frequency_signal": round(self.frequency_signal, 4),
            "finance_density_signal": round(self.finance_density_signal, 4),
            "market_mention_signal": round(self.market_mention_signal, 4),
            "earnings_signal": round(self.earnings_signal, 4),
            "source_credibility": round(self.source_credibility, 4),
            "first_para_signal": round(self.first_para_signal, 4),
            "url_relevance_signal": round(self.url_relevance_signal, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scorer class
# ─────────────────────────────────────────────────────────────────────────────

class FinanceRelevanceScorer:
    """
    High-precision heuristic relevance scoring for financial news articles.
    
    Usage:
        scorer = FinanceRelevanceScorer()
        breakdown = scorer.score(
            title="Reliance Industries Q3 earnings beat expectations",
            text="Full article body text...",
            url="https://economictimes.com/...",
            stock_symbol="RELIANCE",
            aliases=["Reliance Industries", "Reliance Industries Ltd"],
            query_alias_used="Reliance Industries",
        )
        if breakdown.total_score >= 0.50:
            # Article passes high-precision gate
    """
    
    def __init__(
        self,
        high_precision_threshold: float = 0.50,
    ):
        """
        Args:
            high_precision_threshold: Score ≥ this value passes filter (default 0.50)
        """
        self.threshold = high_precision_threshold
    
    def score(
        self,
        title: str,
        text: str,
        url: str,
        stock_symbol: str,
        aliases: List[str],
        query_alias_used: Optional[str] = None,
    ) -> RelevanceScoreBreakdown:
        """
        Compute comprehensive relevance score with full breakdown.
        
        Args:
            title: Article headline
            text: Article body text
            url: Source URL
            stock_symbol: NSE ticker (e.g. "RELIANCE")
            aliases: List of company aliases/search terms
            query_alias_used: Which alias was used in the GDELT query?
        
        Returns:
            RelevanceScoreBreakdown with total_score ∈ [0, 1]
        """
        text_lower = text.lower()
        title_lower = title.lower()
        url_lower = url.lower()
        
        # All searchable terms
        all_terms = [stock_symbol.lower()] + [a.lower() for a in aliases]
        
        # ────────────────────────────────────────────────────────────────────
        # Signal 1: Exact company name in title (strong indicator)
        # ────────────────────────────────────────────────────────────────────
        title_mention = self._score_title_mention(title_lower, all_terms)
        
        # ────────────────────────────────────────────────────────────────────
        # Signal 2: Mention frequency in body text
        # ────────────────────────────────────────────────────────────────────
        frequency_signal = self._score_mention_frequency(text_lower, all_terms)
        
        # ────────────────────────────────────────────────────────────────────
        # Signal 3: Finance keyword density
        # ────────────────────────────────────────────────────────────────────
        finance_density_signal = self._score_finance_density(text_lower)
        
        # ────────────────────────────────────────────────────────────────────
        # Signal 4: Market mention (NSE/BSE/stock market context)
        # ────────────────────────────────────────────────────────────────────
        market_mention_signal = self._score_market_mention(text_lower)
        
        # ────────────────────────────────────────────────────────────────────
        # Signal 5: Earnings/profit signals (highest predictive power)
        # ────────────────────────────────────────────────────────────────────
        earnings_signal = self._score_earnings_mention(text_lower, all_terms)
        
        # ────────────────────────────────────────────────────────────────────
        # Signal 6: Source credibility weighting
        # ────────────────────────────────────────────────────────────────────
        source_credibility = self._score_source_credibility(url_lower)
        
        # ────────────────────────────────────────────────────────────────────
        # Signal 7: Company name in first paragraph
        # ────────────────────────────────────────────────────────────────────
        first_para_signal = self._score_first_paragraph(text_lower, all_terms)
        
        # ────────────────────────────────────────────────────────────────────
        # Signal 8: URL relevance (domain + path keywords)
        # ────────────────────────────────────────────────────────────────────
        url_relevance_signal = self._score_url_relevance(url_lower, all_terms)
        
        # ────────────────────────────────────────────────────────────────────
        # Final scoring (weighted combination)
        # ────────────────────────────────────────────────────────────────────
        total_score = (
            0.25 * title_mention
            + 0.20 * frequency_signal
            + 0.15 * finance_density_signal
            + 0.10 * market_mention_signal
            + 0.10 * earnings_signal
            + 0.10 * source_credibility
            + 0.05 * first_para_signal
            + 0.05 * url_relevance_signal
        )
        
        # Clamp to [0, 1]
        total_score = max(0.0, min(1.0, total_score))
        
        return RelevanceScoreBreakdown(
            total_score=total_score,
            title_mention=title_mention,
            frequency_signal=frequency_signal,
            finance_density_signal=finance_density_signal,
            market_mention_signal=market_mention_signal,
            earnings_signal=earnings_signal,
            source_credibility=source_credibility,
            first_para_signal=first_para_signal,
            url_relevance_signal=url_relevance_signal,
            query_alias_used=query_alias_used,
        )
    
    # ── Individual signal scorers ────────────────────────────────────────────
    
    @staticmethod
    def _score_title_mention(title: str, terms: List[str]) -> float:
        """
        0: Term not in title
        1: Term appears in title
        
        Strong signal: if article is about the company, usually in the headline.
        """
        return 1.0 if any(t in title for t in terms) else 0.0
    
    @staticmethod
    def _score_mention_frequency(text: str, terms: List[str]) -> float:
        """
        Count total mentions; normalize to [0, 1].
        
        ~5+ mentions in a 1000-word article = dedicated coverage.
        """
        total_words = max(len(text.split()), 1)
        mention_count = sum(text.count(t) for t in terms)
        
        # Normalize to mentions per 1000 words
        mentions_per_1k = (mention_count / total_words) * 1000
        
        # Score: 1 mention per 1k words = 0.2, 5+ = 1.0
        score = min(mentions_per_1k / 5.0, 1.0)
        return score
    
    @staticmethod
    def _score_finance_density(text: str) -> float:
        """
        Fraction of sentences containing finance keywords.
        
        High finance density = article is in financial/business context.
        """
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.split()) >= 3]
        
        if not sentences:
            return 0.0
        
        # Combine all finance keywords
        all_finance_kws = EARNINGS_KEYWORDS | MARKET_KEYWORDS | METRICS_KEYWORDS
        
        finance_sents = sum(
            1 for s in sentences
            if any(kw in s for kw in all_finance_kws)
        )
        
        return finance_sents / len(sentences)
    
    @staticmethod
    def _score_market_mention(text: str) -> float:
        """
        Check for explicit market/stock mentions (NSE, BSE, stock, shares).
        
        Binary signal: present or not.
        """
        if any(kw in text for kw in MARKET_KEYWORDS):
            return 1.0
        return 0.0
    
    @staticmethod
    def _score_earnings_mention(text: str, terms: List[str]) -> float:
        """
        Check if earnings/profit mentioned NEAR company name.
        
        This is the highest-value signal for stock relevance.
        Pattern: earnings + company name within 100 chars.
        """
        # Build regex pattern: (earnings keywords) ... (company terms)
        terms_pattern = "|".join(re.escape(t) for t in terms)
        
        for kw in EARNINGS_KEYWORDS:
            # Look for keyword followed by company name within 150 chars
            pattern = rf"{re.escape(kw)}.{{0,150}}({terms_pattern})"
            if re.search(pattern, text, re.IGNORECASE):
                return 1.0
            
            # Also check reverse: company name followed by keyword
            pattern = rf"({terms_pattern}).{{0,150}}{re.escape(kw)}"
            if re.search(pattern, text, re.IGNORECASE):
                return 1.0
        
        return 0.0
    
    @staticmethod
    def _score_source_credibility(url: str) -> float:
        """
        Score URL domain for trusted financial news sources.
        
        Reuters/Bloomberg = 1.0, Economic Times = 0.9, random blog = 0.2
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
        
        # Check trusted sources
        for trusted_domain, weight in TRUSTED_FINANCE_SOURCES.items():
            if trusted_domain in domain:
                return weight
        
        # Check if it's a known financial news domain pattern
        if any(pattern in domain for pattern in ["economictimes", "moneycontrol", "financial"]):
            return 0.7
        
        # Default: lower score for unknown sources
        return 0.3
    
    @staticmethod
    def _score_first_paragraph(text: str, terms: List[str]) -> float:
        """
        Does company name appear in first ~200 words (first paragraph)?
        
        If yes, likely primary subject. If no, might be secondary mention.
        """
        first_para = " ".join(text.split()[:200])
        return 1.0 if any(t in first_para for t in terms) else 0.0
    
    @staticmethod
    def _score_url_relevance(url: str, terms: List[str]) -> float:
        """
        Do URL path/query contain company-relevant keywords?
        
        E.g., /news/reliance/earnings is more relevant than generic /news/page
        """
        url_text = url.lower()
        
        # High-value keywords in URL
        high_value_patterns = {
            "earnings", "results", "profit", "guidance", "outlook",
            "acquisition", "deal", "merger", "ipo",
        }
        
        # Check if URL contains any financial action keyword
        for pattern in high_value_patterns:
            if pattern in url_text:
                return 1.0
        
        # Check if any stock terms appear in URL path
        for term in terms:
            if term.replace(" ", "") in url_text.replace("/", "").replace("-", ""):
                return 0.6
        
        return 0.0


def compute_finance_relevance(
    title: str,
    text: str,
    url: str,
    stock_symbol: str,
    aliases: List[str],
    query_alias: Optional[str] = None,
) -> float:
    """
    Convenience function: compute finance relevance score in one call.
    
    Returns float ∈ [0, 1].
    """
    scorer = FinanceRelevanceScorer()
    breakdown = scorer.score(
        title=title,
        text=text,
        url=url,
        stock_symbol=stock_symbol,
        aliases=aliases,
        query_alias_used=query_alias,
    )
    return breakdown.total_score
