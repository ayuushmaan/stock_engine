"""Source credibility tier mapping for Indian financial news domains.

Tier 1 — Major national business dailies and wire services with
         editorial independence and investigative track record.
Tier 2 — Regional business outlets, portals, and aggregators that
         publish a mix of original and syndicated content.
Tier 3 — PR wires, brand-content studios, and content marketing
         platforms with no editorial firewall.

Used by:
  - pipeline/03_label_news.py   → weak labelling
  - pipeline/04_sponsored_classifier.py → SOURCE_TIER feature
  - models/signal_generator.py  → optional weighting
"""

# tier → set of domain substrings (matched against GDELT SourceURL)
# Matching logic: ``any(domain in source_url for domain in tier_domains)``

TIER1_DOMAINS: list[str] = [
    # ---------- Major English business dailies ----------
    "economictimes.indiatimes.com",
    "livemint.com",
    "business-standard.com",
    "financialexpress.com",
    "moneycontrol.com",
    "ndtvprofit.com",               # formerly ndtv.com/business
    "thehindubusinessline.com",
    # ---------- National broadsheets with business desks ----------
    "thehindu.com",
    "indianexpress.com",
    "hindustantimes.com",
    "ndtv.com",
    "scroll.in",
    "thewire.in",
    # ---------- Wire services ----------
    "reuters.com",
    "bloomberg.com",
    "bbc.com",
    # ---------- Specialist financial data ----------
    "tickertape.in",
    "screener.in",
    "trendlyne.com",
]

TIER2_DOMAINS: list[str] = [
    # ---------- Business / market portals ----------
    "zeebiz.com",
    "cnbctv18.com",
    "firstpost.com",
    "news18.com",
    "outlookbusiness.com",
    "outlookindia.com",
    "businesstoday.in",
    "fortune.com",
    "forbesindia.com",
    "freepressjournal.in",
    "deccanherald.com",
    "deccanchronicle.com",
    "dnaindia.com",
    "theprint.in",
    "swarajyamag.com",
    # ---------- Regional financial sites ----------
    "thequint.com",
    "news9live.com",
    "wionews.com",
    "republicworld.com",
    "timesnownews.com",
    "oneindia.com",
    "jagran.com",
    "amarujala.com",
    # ---------- Market tracking / data sites ----------
    "equitymaster.com",
    "5paisa.com",
    "angelone.in",
    "groww.in",
]

TIER3_DOMAINS: list[str] = [
    # ---------- PR wires ----------
    "prnewswire.com",
    "businesswire.com",
    "globenewswire.com",
    "ani-prsolutions.com",
    "newswire.in",
    "pr.com",
    "einpresswire.com",
    "newsvoir.com",
    "indiaprwire.com",
    "businesswireindia.com",
    # ---------- Content marketing / brand studios ----------
    "brandstudio",          # partial match — catches ET BrandStudio, etc.
    "spotlight",            # many outlets have /spotlight/ sections for native ads
    "partner-content",
    "advertorial",
    # ---------- Known corporate comms domains ----------
    "psuconnect.in",
    "apnlive.com",
    "devdiscourse.com",
    "latestly.com",
    "mybigplunge.com",
]

# ------------------------------------------------------------------
# Flat lookup: domain substring → integer tier
# ------------------------------------------------------------------
DOMAIN_TO_TIER: dict[str, int] = {}
for _domain in TIER1_DOMAINS:
    DOMAIN_TO_TIER[_domain] = 1
for _domain in TIER2_DOMAINS:
    DOMAIN_TO_TIER[_domain] = 2
for _domain in TIER3_DOMAINS:
    DOMAIN_TO_TIER[_domain] = 3


def classify_source(source_url: str) -> int:
    """Return credibility tier (1/2/3) for a source URL.

    Matching is done by substring containment against the domain list.
    If no match is found, returns 2 (unknown → mid-tier default).

    Parameters
    ----------
    source_url : str
        Full URL or domain string from GDELT SourceURL / SourceCommonName.

    Returns
    -------
    int
        1, 2, or 3 indicating credibility tier.
    """
    url_lower = source_url.lower()
    for domain, tier in DOMAIN_TO_TIER.items():
        if domain in url_lower:
            return tier
    return 2   # default: treat unknown sources as mid-tier


def is_pr_wire(source_url: str) -> bool:
    """Return True if the source URL belongs to a known PR wire domain."""
    url_lower = source_url.lower()
    return any(d in url_lower for d in TIER3_DOMAINS[:10])  # first 10 are PR wires


def is_tier1(source_url: str) -> bool:
    """Return True if the source URL belongs to a Tier-1 outlet."""
    return classify_source(source_url) == 1


# ------------------------------------------------------------------
# Quality outlets used for organic weak-label generation
# Articles with negative/neutral tone from these are "definite organic"
# ------------------------------------------------------------------
ORGANIC_LABEL_DOMAINS: list[str] = [
    "economictimes.indiatimes.com",
    "livemint.com",
    "business-standard.com",
    "thehindu.com",
    "moneycontrol.com",
    "ndtvprofit.com",
    "thehindubusinessline.com",
    "reuters.com",
    "bloomberg.com",
    "indianexpress.com",
]
