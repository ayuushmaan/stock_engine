"""Observable news morphology and structural feature extractors.

All features are deterministic functions of observable document attributes
(URL path, domain, headline, token morphology, and boilerplate structure).
Features are computed without using sentiment scores or circular labeling heuristics.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Any, Dict, List, Optional, Union

# ── Explicit statutory & platform sponsored keywords / path tokens ───
EXPLICIT_SPONSORED_PATHS = [
    # Top mainstream newspaper PR syndication paths (India & Global)
    "press-releases-ani",
    "/press-releases-ani/",
    "/content/press-releases-ani/",
    "press-releases-ptipr",
    "/ptipr/",
    "/pti-pr/",
    "/press-release/",
    "/press-releases/",
    "/pressrelease/",
    "/pressreleases/",
    "/news-releases/",
    "/news-release/",
    "/pnn/",
    "/pnn-breaking-news/",
    "/mediawire/",
    "/businesswireindia/",
    "/pr-newswire/",
    "/newsvoir/",
    # Brand studio / Advertorial paths
    "/brandstudio/",
    "/brandstudio",
    "/brand-studio/",
    "/brand-studio",
    "/spotlight/",
    "/spotlight",
    "/advertorial/",
    "/advertorial",
    "/advertorials/",
    "/sponsored/",
    "/sponsored-content/",
    "/brand-connect/",
    "/brand_connect/",
    "/brandconnect/",
    "/partner-content/",
    "/partner_content/",
    "/partnered-content/",
    "/content-marketing/",
    "/native-content/",
    "/brand-stories/",
    "/brand-post/",
    "/brand-voice/",
    "/impact-feature/",
]

# PR Wire domains that provide guaranteed paid commercial distribution
PR_WIRE_DOMAINS = {
    "prnewswire.com",
    "businesswire.com",
    "globenewswire.com",
    "newsvoir.com",
    "einpresswire.com",
    "indiaprwire.com",
    "businesswireindia.com",
    "ani-prsolutions.com",
    "newswire.in",
    "pr.com",
    "openpr.com",
    "prlog.org",
    "mediawire.in",
    "pnn.co.in",
    "apnlive.com",
    "devdiscourse.com",
    "latestly.com",
    "mybigplunge.com",
    "psuconnect.in",
}

# Market research PR mills that spam stock tickers to sell $4,000 PDF reports
MARKET_RESEARCH_SPAM_FIRMS = [
    "the-insight-partners",
    "the insight partners",
    "allied-market-research",
    "allied market research",
    "grand-view-research",
    "grand view research",
    "fortune-business-insights",
    "fortune business insights",
    "transparency-market-research",
    "transparency market research",
    "marketsandmarkets",
    "coherent-market-insights",
    "market-research-future",
    "custom-market-insights",
    "verified-market-research",
]

MARKET_RESEARCH_SPAM_PATTERNS = [
    r"-market-to-reach-",
    r"-market-size-",
    r"-market-growth-",
    r"-market-forecast-",
    r"\bmarket\s+to\s+reach\s+(?:usd|us\$|rs|\$)\b",
    r"\bgrowth\s+at\s+(?:a\s+)?cagr\s+of\b",
    r"\bmarket\s+size\s+(?:is\s+)?expected\s+to\s+reach\b",
    r"\bkey\s+players\s+(?:profiled|operating\s+in|include)\b",
]

# Statutory and commercial disclosure phrases in text or bylines
DISCLOSURE_PATTERNS = [
    r"\bsponsored\s+feature\b",
    r"\bpaid\s+promotional\s+content\b",
    r"\bpartnered\s+content\b",
    r"\bcontent\s+created\s+by\s+brand\b",
    r"\bviews\s+expressed\s+are\s+(?:personal|those\s+of\s+the\s+author)\s+and\s+not\s+of\s+(?:the\s+)?editorial\b",
    r"\bthis\s+is\s+a\s+sponsored\s+(?:post|article|story)\b",
    r"\bpresented\s+by\b",
    r"\bbrand\s+connect\s+initiative\b",
    r"\badvertorial\b",
    r"\bmediawire\s+initiative\b",
    r"\bpress\s+release\s+distributed\s+by\b",
]

# Curated promotional superlative n-grams
PROMO_KEYWORDS = [
    "record-breaking",
    "best-ever",
    "strong growth",
    "milestone",
    "proud to announce",
    "pleased to share",
    "delighted to announce",
    "groundbreaking",
    "revolutionary",
    "industry-leading",
    "award-winning",
    "stellar performance",
    "transformative",
    "state-of-the-art",
    "pioneering",
    "exponential growth",
    "visionary leadership",
    "multibagger",
    "hidden gem",
    "rocket stock",
    "ready to surge",
    "skyrockets",
]

# Commercial call-to-action (CTA) patterns
CTA_PATTERNS = [
    r"\bvisit\s+(?:https?:\/\/|www\.)[^\s]+",
    r"\bclick\s+here\s+to\s+(?:buy|register|download|learn|visit|book|invest)\b",
    r"\bbook\s+(?:your\s+)?(?:slot|order|seat|demo|appointment)\b",
    r"\bavailable\s+on\s+(?:amazon|flipkart|app\s+store|google\s+play)\b",
    r"\bdownload\s+the\s+(?:app|whitepaper|report|sample)\b",
    r"\bfor\s+(?:more\s+details|inquiries|sales),\s*(?:contact|visit|call|email)\b",
    r"\blimited\s+time\s+offer\b",
    r"\buse\s+(?:coupon|code|promo)\b",
    r"\bapply\s+now\b",
    r"\bregister\s+today\b",
    r"\bget\s+sample\s+pdf\s+copy\b",
]

# Risk, dispute, and critical editorial terminology
RISK_KEYWORDS = [
    "fraud",
    "penalty",
    "lawsuit",
    "litigation",
    "recall",
    "default",
    "downgrade",
    "regulatory probe",
    "investigation",
    "write-off",
    "insolvency",
    "violation",
    "scam",
    "debt distress",
    "margin compression",
    "headwinds",
    "underperformance",
    "misses estimates",
    "loss widened",
]

# Boilerplate footer patterns common in corporate press releases
BOILERPLATE_PATTERNS = [
    r"\babout\s+[A-Z][A-Za-z0-9\s,\.]+(?:ltd|limited|inc|corp|corporation|technologies|solutions)?\s*:",
    r"\bmedia\s+contact\s*:",
    r"\bfor\s+media\s+inquiries\s*:",
    r"\bforward-looking\s+statements\b",
    r"\bsafe\s+harbor\s+statement\b",
    r"\bsource\s*:\s*[A-Z][A-Za-z0-9\s,\.]+",
]


def extract_url_sponsored_flag(url: Optional[str]) -> float:
    """Check if URL path or query contains explicit sponsored tokens.
    
    Returns 1.0 if match, 0.0 otherwise.
    """
    if not url or not isinstance(url, str):
        return 0.0
    url_lower = url.lower()
    for pat in EXPLICIT_SPONSORED_PATHS:
        if pat in url_lower:
            return 1.0
    return 0.0


def extract_market_research_spam_flag(url_or_text: Optional[str]) -> float:
    """Detect commercial market research report spam that syndicates ticker names.
    
    Returns 1.0 if match, 0.0 otherwise.
    """
    if not url_or_text or not isinstance(url_or_text, str):
        return 0.0
    target_lower = url_or_text.lower()
    
    # Check known market research spam companies
    for firm in MARKET_RESEARCH_SPAM_FIRMS:
        if firm in target_lower:
            return 1.0
            
    # Check market forecast patterns
    for pat in MARKET_RESEARCH_SPAM_PATTERNS:
        if re.search(pat, target_lower):
            return 1.0
            
    return 0.0


def extract_pr_wire_flag(url_or_domain: Optional[str]) -> float:
    """Check if domain or URL matches a known commercial PR wire domain.
    
    Returns 1.0 if match, 0.0 otherwise.
    """
    if not url_or_domain or not isinstance(url_or_domain, str):
        return 0.0
    domain = url_or_domain.lower()
    for pr_dom in PR_WIRE_DOMAINS:
        if pr_dom in domain:
            return 1.0
    return 0.0


def extract_disclosure_flag(text: Optional[str]) -> float:
    """Scan text for statutory/commercial disclaimer markers.
    
    Returns 1.0 if match, 0.0 otherwise.
    """
    if not text or not isinstance(text, str):
        return 0.0
    for pat in DISCLOSURE_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return 1.0
    return 0.0


def extract_promo_density(text: Optional[str], gamma_promo: float = 50.0) -> float:
    """Calculate normalized density of promotional superlative phrases in text.
    
    Formula: min(1.0, (promo_hits / total_tokens) * gamma_promo)
    """
    if not text or not isinstance(text, str):
        return 0.0
    tokens = re.findall(r"\b\w+\b", text.lower())
    n_tokens = len(tokens)
    if n_tokens == 0:
        return 0.0

    text_lower = text.lower()
    hits = sum(text_lower.count(kw) for kw in PROMO_KEYWORDS)
    raw_density = hits / max(n_tokens, 20)
    return float(min(1.0, raw_density * gamma_promo))


def extract_cta_density(text: Optional[str], gamma_cta: float = 25.0) -> float:
    """Calculate density of commercial call-to-action imperatives in text.
    
    Formula: min(1.0, (cta_regex_hits / total_tokens) * gamma_cta)
    """
    if not text or not isinstance(text, str):
        return 0.0
    tokens = re.findall(r"\b\w+\b", text)
    n_tokens = len(tokens)
    if n_tokens == 0:
        return 0.0

    hits = 0
    for pat in CTA_PATTERNS:
        hits += len(re.findall(pat, text, flags=re.IGNORECASE))
    
    raw_density = hits / max(n_tokens, 20)
    return float(min(1.0, raw_density * gamma_cta))


def extract_boilerplate_ratio(text: Optional[str]) -> float:
    """Calculate ratio of boilerplate/media contact section to total article length.
    
    Finds the earliest occurrence of 'About [Company]' or 'Media Contact:' and
    computes the trailing token fraction.
    """
    if not text or not isinstance(text, str):
        return 0.0
    tokens = re.findall(r"\b\w+\b", text)
    n_tokens = len(tokens)
    if n_tokens < 30:
        return 0.0

    earliest_idx = len(text)
    matched = False
    for pat in BOILERPLATE_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m and m.start() < earliest_idx:
            earliest_idx = m.start()
            matched = True

    if not matched:
        return 0.0

    boilerplate_text = text[earliest_idx:]
    boilerplate_tokens = len(re.findall(r"\b\w+\b", boilerplate_text))
    return float(min(1.0, max(0.0, boilerplate_tokens / n_tokens)))


def extract_risk_absence(text: Optional[str], lambda_risk: float = 40.0) -> float:
    """Calculate risk absence score.
    
    If risk tokens are zero -> score = 1.0 (typical of PR).
    As risk tokens increase -> score decays toward 0.0 (typical of organic journalism).
    Formula: exp(-lambda_risk * (risk_hits / total_tokens))
    """
    if not text or not isinstance(text, str):
        return 1.0
    tokens = re.findall(r"\b\w+\b", text.lower())
    n_tokens = len(tokens)
    if n_tokens == 0:
        return 1.0

    text_lower = text.lower()
    hits = sum(text_lower.count(kw) for kw in RISK_KEYWORDS)
    ratio = hits / max(n_tokens, 20)
    import math
    return float(math.exp(-lambda_risk * ratio))


def extract_headline_hype(headline: Optional[str]) -> float:
    """Calculate headline hype index based on superlatives and punctuation entropy.
    
    Formula: (superlative_count / headline_tokens) + 0.5 * I('!' or '?' in headline)
    """
    if not headline or not isinstance(headline, str):
        return 0.0
    tokens = re.findall(r"\b\w+\b", headline.lower())
    n_tokens = len(tokens)
    if n_tokens == 0:
        return 0.0

    headline_lower = headline.lower()
    superlative_hits = sum(headline_lower.count(kw) for kw in PROMO_KEYWORDS)
    punct_flag = 0.5 if ("!" in headline or "?" in headline) else 0.0
    
    raw_score = (superlative_hits / max(n_tokens, 3)) + punct_flag
    return float(min(1.0, max(0.0, raw_score)))


def extract_quote_monopoly(text: Optional[str]) -> float:
    """Calculate single-source executive quote dominance ratio.
    
    Approximates the fraction of quotes originating from corporate management.
    """
    if not text or not isinstance(text, str):
        return 0.5
    quotes = re.findall(r'"([^"]*)"', text)
    if not quotes:
        return 0.5  # Neutral default when no quotes are present
    
    exec_cues = ["said mr", "said ms", "stated", "managing director", "ceo", "president", "commented"]
    text_lower = text.lower()
    exec_quote_hits = sum(1 for cue in exec_cues if cue in text_lower)
    
    return float(min(1.0, (exec_quote_hits + 0.5) / (len(quotes) + 1.0)))


def extract_all_morphometrics(
    url: Optional[str] = None,
    headline: Optional[str] = None,
    text: Optional[str] = None,
    domain: Optional[str] = None,
) -> Dict[str, float]:
    """Extract complete observable feature vector for an article.
    
    Returns a dictionary of normalized float values in [0, 1].
    """
    target_text = text if (text and len(text.strip()) > 0) else (url or "")
    target_domain = domain or (urllib.parse.urlparse(url).netloc if url else "")
    target_headline = headline or ""

    return {
        "pr_wire_flag": extract_pr_wire_flag(target_domain or url),
        "url_sponsor_flag": extract_url_sponsored_flag(url),
        "market_research_spam_flag": extract_market_research_spam_flag(url or target_text),
        "disclosure_flag": extract_disclosure_flag(target_text),
        "promo_density": extract_promo_density(target_text),
        "cta_density": extract_cta_density(target_text),
        "boilerplate_ratio": extract_boilerplate_ratio(target_text),
        "risk_absence": extract_risk_absence(target_text),
        "headline_hype": extract_headline_hype(target_headline),
        "quote_monopoly": extract_quote_monopoly(target_text),
    }
