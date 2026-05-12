"""
migrate_articles_schema.py
─────────────────────────────────────────────────────────────────────────────
Idempotent migration: adds new columns introduced by historical_news_collector
to the existing `articles` table.

Run once after STEP 1:
    python migrate_articles_schema.py --db stock_engine.db

Columns added:
  source_credibility  REAL     — domain credibility weight [0, 1]
  session_lag         TEXT     — PRE_MARKET | INTRADAY | POST_MARKET
  relevance_score     REAL     — entity relevance [0, 1]
  finance_density     REAL     — fraction of sentences with finance keywords
  word_count          INTEGER  — article word count
  gdelt_tone          REAL     — raw GDELT composite tone [-100, 100]
─────────────────────────────────────────────────────────────────────────────
"""

import sqlite3
import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "db" / "stock_engine.db"

NEW_COLUMNS = [
    ("source_credibility", "REAL"),
    ("session_lag",        "TEXT"),
    ("relevance_score",    "REAL"),
    ("finance_density",    "REAL"),
    ("word_count",         "INTEGER"),
    ("gdelt_tone",         "REAL"),
]


def migrate(db_path: str, verbose: bool = True):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get existing columns
    cursor.execute("PRAGMA table_info(articles)")
    existing = {row[1] for row in cursor.fetchall()}

    added = []
    for col_name, col_type in NEW_COLUMNS:
        if col_name not in existing:
            cursor.execute(f"ALTER TABLE articles ADD COLUMN {col_name} {col_type}")
            added.append(col_name)
            logger.info(f"  Added: {col_name} {col_type}")

    # Add a composite index to accelerate relevance filtering
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_articles_relevance
        ON articles (stock_symbol, collection_date, relevance_score)
    """)

    # Add index for session_lag (useful in calibration: filter by timing)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_articles_session
        ON articles (stock_symbol, session_lag)
    """)

    conn.commit()
    conn.close()

    if verbose:
        if added:
            print(f"Migration complete. Added {len(added)} columns: {added}")
        else:
            print("Already up-to-date. No changes needed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()
    migrate(args.db)
