"""
Configuration file for NIFTY 50 Stock Prediction Engine
Modify these settings to customize behavior
"""

# ═══════════════════════════════════════════════════════════════════
# TIME WINDOW CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Valid news collection window (24-hour format IST)
NEWS_WINDOW_START = 16  # 4:00 PM (after market close)
NEWS_WINDOW_END = 9     # 9:00 AM (before market open)
TIMEZONE = 'Asia/Kolkata'

# Enable time filtering (set False to disable for testing)
ENABLE_TIME_FILTERING = True

# ═══════════════════════════════════════════════════════════════════
# SENTIMENT ANALYSIS CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# FinBERT model to use for sentiment analysis
FINBERT_MODEL = "ProsusAI/finbert"

# Sentiment score threshold for signal generation
# Scores beyond ±SENTIMENT_THRESHOLD are considered significant
SENTIMENT_THRESHOLD = 0.05

# BULLISH if score > SENTIMENT_THRESHOLD
# BEARISH if score < -SENTIMENT_THRESHOLD
# NEUTRAL if -SENTIMENT_THRESHOLD <= score <= SENTIMENT_THRESHOLD

# ═══════════════════════════════════════════════════════════════════
# NEWS SOURCES CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Google News configuration
NEWS_LANGUAGE = 'en'
NEWS_REGION = 'IN'
NEWS_PERIOD = '1d'  # Last 1 day

# Search query suffix for company news
NEWS_SEARCH_SUFFIX = "share news"  # E.g., "INFY share news"

# ═══════════════════════════════════════════════════════════════════
# STOCK UNIVERSE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Index to scan: NIFTY 50, NIFTY 100, NIFTY 200, etc.
INDEX_NAME = "NIFTY%2050"
INDEX_URL = f"https://www.nseindia.com/api/equity-stockIndices?index={INDEX_NAME}"

# For testing, you can sample companies:
# Set to None to scan all companies
# Set to any integer N to randomly sample N companies
SAMPLE_SIZE = None  # None means scan all available

# ═══════════════════════════════════════════════════════════════════
# DATA FETCHING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Stock data source
STOCK_DATA_PERIOD = "1d"  # Last 1 day
STOCK_DATA_INTERVAL = "1d"  # Daily interval

# Data output directory
DATA_OUTPUT_DIR = "data"

# CSV output files
CSV_DAILY_PRICES = "daily_prices.csv"
CSV_SENTIMENT_RESULTS = "nifty_sentiment_results.csv"
CSV_PREDICTION_REPORT = "prediction_report.csv"

# ═══════════════════════════════════════════════════════════════════
# SCHEDULING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Automatic run times (HH:MM format, 24-hour IST)
SCHEDULE_TIMES = [
    "00:00",  # Midnight - start of 4pm-9am window
    "08:00",  # 8am - before market open (last hour of window)
]

# Run continuously in background (True) or one-time (False)
CONTINUOUS_SCHEDULING = True

# ═══════════════════════════════════════════════════════════════════
# ERROR HANDLING & LOGGING
# ═══════════════════════════════════════════════════════════════════

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# Logging
LOG_TO_FILE = True
LOG_FILE = "stock_engine.log"
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# ═══════════════════════════════════════════════════════════════════
# PERFORMANCE TUNING
# ═══════════════════════════════════════════════════════════════════

# Multi-threading for faster news fetching
ENABLE_MULTITHREADING = False
MAX_THREADS = 4

# Progress bar display
SHOW_PROGRESS_BAR = True

# ═══════════════════════════════════════════════════════════════════
# FEATURE FLAGS (for future enhancements)
# ═══════════════════════════════════════════════════════════════════

# Database persistence
USE_DATABASE = False
DATABASE_PATH = "stock_predictions.db"

# Email notifications
SEND_EMAIL_ALERTS = False
EMAIL_RECIPIENTS = ["your_email@example.com"]

# Webhook integration with external services
ENABLE_WEBHOOKS = False
WEBHOOK_URL = ""

# ═══════════════════════════════════════════════════════════════════
# ADVANCED SETTINGS
# ═══════════════════════════════════════════════════════════════════

# Adjust sentiment score calculation
# If True: Average all article scores
# If False: Weight by recency
SENTIMENT_WEIGHTING = "average"

# Minimum number of news articles required for sentiment analysis
MIN_ARTICLES_FOR_ANALYSIS = 1

# Filter out news older than X hours
MAX_NEWS_AGE_HOURS = 24

# ═══════════════════════════════════════════════════════════════════
# HELP & DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════

"""
HOW TO USE THIS CONFIG FILE:

1. Read the comments above each setting
2. Modify values as needed for your use case
3. Most users should leave default settings unchanged

EXAMPLE CUSTOMIZATIONS:

   a) Test with 10 companies first:
      Set: SAMPLE_SIZE = 10

   b) Disable time window checking:
      Set: ENABLE_TIME_FILTERING = False

   c) Increase sensitivity (lower threshold):
      Set: SENTIMENT_THRESHOLD = 0.03

   d) Use different news source region:
      Set: NEWS_REGION = 'US'

For more details, see README.md
"""

# ═══════════════════════════════════════════════════════════════════
# END OF CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
