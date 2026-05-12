"""
Database Schema Management for Continuous Stock Prediction Engine

Handles SQLite table creation, migrations, and schema initialization.
"""

import sqlite3
import pytz

from SRC.common.paths import DB_DIR, ensure_dirs

class DatabaseSchema:
    """Manages SQLite database schema for stock prediction engine"""
    
    def __init__(self, db_path: str = str(DB_DIR / "stock_engine.db")):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.ist = pytz.timezone('Asia/Kolkata')
        
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_schema(self, verbose: bool = True):
        """Create all tables if they don't exist"""
        ensure_dirs()
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Table 1: Articles
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS articles (
                    article_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_date DATE NOT NULL,
                    stock_symbol TEXT NOT NULL,
                    
                    title TEXT NOT NULL,
                    headline_url TEXT,
                    article_text TEXT,
                    published_time TIMESTAMP,
                    fetched_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Metadata
                    is_sponsored INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'GoogleNews',
                    
                    -- Sentiment scores (FinBERT output)
                    sentiment_score REAL,
                    sentiment_positive_prob REAL,
                    sentiment_negative_prob REAL,
                    sentiment_neutral_prob REAL,
                    
                    UNIQUE(stock_symbol, title, collection_date)
                )
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_articles_date_symbol 
                ON articles(collection_date, stock_symbol)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_articles_sponsored 
                ON articles(is_sponsored)
            ''')
            
            # Table 2: Daily Signals (Aggregated Stock-Level Predictions)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_signals (
                    signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_date DATE NOT NULL,
                    stock_symbol TEXT NOT NULL,
                    
                    -- Article counts
                    article_count INTEGER DEFAULT 0,
                    sponsored_count INTEGER DEFAULT 0,
                    organic_count INTEGER DEFAULT 0,
                    
                    -- Raw aggregated sentiment (before normalization)
                    all_sentiment_raw REAL,
                    sponsored_sentiment_raw REAL,
                    organic_sentiment_raw REAL,
                    
                    -- Normalized prediction scores [-1, 1]
                    pred_score REAL,
                    sponsored_pred_score REAL,
                    organic_pred_score REAL,
                    
                    -- Direction classification
                    pred_direction TEXT,
                    direction_threshold REAL DEFAULT 0.1,
                    
                    -- Expected % move
                    expected_pct_change REAL,
                    calibration_k REAL,
                    
                    -- Metadata
                    processing_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    is_production INTEGER DEFAULT 1,
                    normalization_alpha REAL,
                    
                    UNIQUE(collection_date, stock_symbol)
                )
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_signals_date_symbol 
                ON daily_signals(collection_date, stock_symbol)
            ''')
            
            # Table 3: Daily Prices
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_prices (
                    price_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    stock_symbol TEXT NOT NULL,
                    
                    open_price REAL,
                    close_price REAL,
                    high_price REAL,
                    low_price REAL,
                    volume INTEGER,
                    
                    -- Actual movement
                    actual_pct_change REAL,
                    actual_direction TEXT,
                    
                    -- Metadata
                    fetched_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    data_source TEXT DEFAULT 'yfinance',
                    
                    UNIQUE(date, stock_symbol)
                )
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_prices_date_symbol 
                ON daily_prices(date, stock_symbol)
            ''')
            
            # Table 4: Evaluations (Performance Metrics)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evaluations (
                    eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    eval_date DATE NOT NULL,
                    lookback_days INTEGER NOT NULL,
                    
                    -- Correlation metrics
                    pearson_corr REAL,
                    spearman_corr REAL,
                    
                    -- Error metrics
                    mae REAL,
                    rmse REAL,
                    mape REAL,
                    
                    -- Direction accuracy
                    directional_accuracy REAL,
                    
                    -- Subset metrics
                    sponsored_pearson_corr REAL,
                    organic_pearson_corr REAL,
                    sponsored_sample_count INTEGER,
                    organic_sample_count INTEGER,
                    
                    -- Data
                    sample_count INTEGER DEFAULT 0,
                    
                    created_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(eval_date, lookback_days)
                )
            ''')
            
            # Table 5: Calibration History
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS calibration_history (
                    cal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    calibration_date DATE NOT NULL,
                    lookback_days INTEGER NOT NULL,
                    
                    -- Learned parameters
                    k_optimal REAL NOT NULL,
                    threshold_bullish REAL DEFAULT 0.1,
                    threshold_bearish REAL DEFAULT -0.1,
                    normalization_param REAL DEFAULT 1.5,
                    
                    -- Fit quality
                    rmse_on_train REAL,
                    rmse_on_test REAL,
                    pearson_on_train REAL,
                    pearson_on_test REAL,
                    
                    -- Metadata
                    is_active INTEGER DEFAULT 1,
                    train_sample_count INTEGER,
                    test_sample_count INTEGER,
                    created_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(calibration_date, lookback_days)
                )
            ''')
            
            # Table 6: Run Log (Idempotency tracking)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS run_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date DATE NOT NULL,
                    run_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    
                    stocks_processed INTEGER,
                    articles_collected INTEGER,
                    
                    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    
                    UNIQUE(run_date, run_type)
                )
            ''')
            
            conn.commit()
            if verbose:
                print(f"[OK] Database schema initialized: {self.db_path}")
            return True
            
        except Exception as e:
            conn.rollback()
            if verbose:
                print(f"[ERROR] Error initializing schema: {e}")
            return False
        finally:
            conn.close()
    
    def get_schema_info(self):
        """Print current schema information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print("\n" + "="*70)
        print("DATABASE SCHEMA INFO")
        print("="*70)
        print(f"Database: {self.db_path}")
        print(f"Tables: {len(tables)}")
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            print(f"\n  {table_name} ({len(columns)} columns)")
            for col in columns:
                print(f"    - {col[1]}: {col[2]}")
        
        print("\n" + "="*70 + "\n")
        conn.close()
    
    def clear_test_data(self, run_date):
        """Clear test data from a specific date (for idempotent reruns)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Delete from daily_signals where is_production=0
            cursor.execute(
                "DELETE FROM daily_signals WHERE collection_date = ? AND is_production = 0",
                (run_date,)
            )
            
            # Delete articles from that date
            cursor.execute(
                "DELETE FROM articles WHERE collection_date = ? AND is_sponsored = 0",
                (run_date,)
            )
            
            conn.commit()
            print(f"[OK] Test data cleared for {run_date}")
            return True
        except Exception as e:
            conn.rollback()
            print(f"[ERROR] Error clearing test data: {e}")
            return False
        finally:
            conn.close()
    
    def get_stats(self):
        """Get database statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        try:
            cursor.execute("SELECT COUNT(*) FROM articles")
            stats['total_articles'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM articles WHERE is_sponsored = 0")
            stats['organic_articles'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM daily_signals")
            stats['total_signals'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM daily_prices")
            stats['total_prices'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM evaluations")
            stats['total_evaluations'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM calibration_history WHERE is_active = 1")
            stats['active_calibrations'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT stock_symbol) FROM articles")
            stats['unique_stocks_covered'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(collection_date), MAX(collection_date) FROM articles")
            result = cursor.fetchone()
            stats['date_range'] = (result[0], result[1]) if result[0] else ("N/A", "N/A")
            
            return stats
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}
        finally:
            conn.close()


if __name__ == "__main__":
    # Initialize database
    db = DatabaseSchema()
    db.init_schema()
    db.get_schema_info()
    
    # Print initial stats
    print("Current Database Statistics:")
    stats = db.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
