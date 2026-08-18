"""Central configuration for NIFTY 50 News-Signal Research Project.

All tunable parameters, date ranges, thresholds, and constants live here.
No side-effects on import — call ``seed_everything()`` explicitly in scripts.
"""
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_FINAL = PROJECT_ROOT / "data" / "final"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_TABLES = OUTPUTS_DIR / "tables"
OUTPUTS_FIGURES = OUTPUTS_DIR / "figures"

# Sub-directories inside raw
RAW_GDELT = DATA_RAW / "gdelt"
RAW_PRICES = DATA_RAW / "prices"

# Ensure directories exist on first import
for _d in [
    RAW_GDELT, RAW_PRICES,
    DATA_PROCESSED, DATA_FINAL,
    MODELS_DIR,
    OUTPUTS_TABLES, OUTPUTS_FIGURES,
]:
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATE SPLITS  (no look-ahead bias)
# ============================================================
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-12-31"
VALIDATION_START = "2024-01-01"
VALIDATION_END = "2024-06-30"
HOLDOUT_START = "2024-07-01"
HOLDOUT_END = "2026-03-31"

# Full data range for price & GDELT fetching
FETCH_START = "2019-12-01"
FETCH_END = "2026-03-31"

# ============================================================
# MARKET PARAMETERS  (IST timezone)
# ============================================================
MARKET_OPEN_IST = "09:15"
MARKET_CLOSE_IST = "15:30"
TIMEZONE = "Asia/Kolkata"
NSE_SUFFIX = ".NS"

# Time-bucket boundary strings (HH:MM in IST)
CLOSED_POST_START = "15:30"  # market close
CLOSED_POST_END = "23:59"    # midnight
CLOSED_PRE_START = "00:00"   # midnight
CLOSED_PRE_END = "09:15"     # market open
OPEN_START = "09:15"
OPEN_END = "15:30"

# ============================================================
# SIGNAL GENERATOR PARAMETERS
# ============================================================
ALPHA = 1.8                       # tanh scaling factor
DIRECTION_THRESHOLD = 0.10        # |pred_score| > this → BULLISH/BEARISH
ORGANIC_WEIGHT_FLOOR = 0.3        # minimum organic weight
TIME_WEIGHT_CLOSED = 1.5          # weight multiplier for closed-window news
TIME_WEIGHT_OPEN = 1.0            # weight multiplier for open-window news
SPONSORED_PROB_HIGH = 0.7         # above this → "sponsored"
SPONSORED_PROB_LOW = 0.3          # below this → "organic"
MIN_ARTICLES_FOR_SIGNAL = 3       # minimum articles to form a daily signal
TONE_NORMALIZER = 10.0            # divide GDELT tone by this to get ≈[-1,1]

# ============================================================
# CLASSIFIER PARAMETERS  (LightGBM)
# ============================================================
CLASSIFIER_TRAIN_START = "2020-01-01"
CLASSIFIER_TRAIN_END = "2022-12-31"
CLASSIFIER_VALID_START = "2023-01-01"
CLASSIFIER_VALID_END = "2023-12-31"

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
}

# ============================================================
# KEYWORD LISTS FOR FEATURE ENGINEERING
# ============================================================
PROMO_KEYWORDS = [
    "record", "best-ever", "strong growth", "robust", "milestone",
    "beats", "exceeds", "outperforms", "proud to announce",
    "pleased to share", "honored", "delighted to", "landmark",
    "strong", "proud", "delighted", "pleased",
]

RISK_KEYWORDS = [
    "fraud", "loss", "penalty", "lawsuit", "recall", "default",
    "downgrade", "misses", "below expectations", "regulatory",
    "probe", "investigation", "writeoff", "insolvency",
    "miss", "scam", "violation",
]

# ============================================================
# SPONSORED-CONTENT URL & DOMAIN PATTERNS
# ============================================================
SPONSORED_URL_PATTERNS = [
    "/brandstudio/", "/spotlight/", "/advertorial/", "/sponsored/",
    "/brand-connect/", "/prime/", "/partner/", "/branded-content/",
    "/content-marketing/", "/native-content/", "/partner-content/",
]

PR_WIRE_DOMAINS = [
    "prnewswire.com", "businesswire.com", "globenewswire.com",
    "ani-prsolutions.com", "newswire.in", "pr.com", "einpresswire.com",
    "newsvoir.com", "indiaprwire.com", "businesswireindia.com",
]

# ============================================================
# MARKET-REGIME THRESHOLDS
# ============================================================
MARKET_REGIME_LOOKBACK_DAYS = 60          # rolling window
BULL_THRESHOLD = 0.10                      # 60-day return >  +10 %
BEAR_THRESHOLD = -0.10                     # 60-day return < -10 %
NIFTY50_INDEX_TICKER = "^NSEI"            # Yahoo Finance symbol for Nifty 50

# Market-cap tiers for robustness checks
CAP_TIER_TOP10 = 10
CAP_TIER_MID = 25
# stocks 1-10 → tier "TOP10", 11-25 → "MID", 26-50 → "TAIL"

# ============================================================
# RESEARCH PARAMETERS
# ============================================================
IC_WINDOW_DAYS = 60              # rolling window for IC calculation
BOOTSTRAP_SAMPLES = 5000         # 2×2 interaction bootstrap
EVENT_STUDY_WINDOW = (-2, 5)     # days around event for CAR
QUINTILE_THRESHOLD = 0.80        # top quintile for event definition

# ============================================================
# BIGQUERY / GDELT
# ============================================================
# Set your GCP project ID here or via env var GOOGLE_CLOUD_PROJECT
GCP_PROJECT_ID = "stock-sentiment-app-315d6"

GDELT_TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"
GDELT_EVENTS_TABLE = "gdelt-bq.gdeltv2.events"

# Maximum bytes billed per BigQuery query (safety guard)
MAX_BYTES_BILLED = 30 * 1024 ** 3   # 30 GB per query

# ============================================================
# RANDOM SEEDS
# ============================================================
NUMPY_SEED = 42
PYTHON_SEED = 42
LIGHTGBM_SEED = 42


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility. Call this at the top of
    every pipeline script and research notebook."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# LOGGING CONFIGURATION
# ============================================================
import logging  # noqa: E402

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger and return a named logger for the caller."""
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
    return logging.getLogger("nifty_research")