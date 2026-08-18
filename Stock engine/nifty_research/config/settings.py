from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# Date splits
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-12-31"
VALIDATION_START = "2024-01-01"
VALIDATION_END = "2024-06-30"
HOLDOUT_START = "2024-07-01"
HOLDOUT_END = "2026-03-31"

# Market hours (IST)
MARKET_OPEN_IST = "09:15"
MARKET_CLOSE_IST = "15:30"
TIMEZONE = "Asia/Kolkata"
NSE_SUFFIX = ".NS"

# Signal generator parameters
ALPHA = 1.8
DIRECTION_THRESHOLD = 0.10
ORGANIC_WEIGHT_FLOOR = 0.3
TIME_WEIGHT_CLOSED = 1.5
TIME_WEIGHT_OPEN = 1.0
SPONSORED_PROB_HIGH = 0.7
SPONSORED_PROB_LOW = 0.3
MIN_ARTICLES_FOR_SIGNAL = 3

# Keyword lists for sponsored/organic detection
PROMO_KEYWORDS = [
    "record", "best-ever", "strong growth", "robust", "milestone",
    "beats", "exceeds", "outperforms", "proud to announce",
    "pleased to share", "honored", "delighted to", "landmark"
]

RISK_KEYWORDS = [
    "fraud", "loss", "penalty", "lawsuit", "recall", "default",
    "downgrade", "misses", "below expectations", "regulatory",
    "probe", "investigation", "writeoff", "insolvency"
]

# Random seeds
NUMPY_SEED = 42
PYTHON_SEED = 42
LIGHTGBM_SEED = 42