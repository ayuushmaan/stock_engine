#!/usr/bin/env python3
"""
Initialization script for continuous stock prediction engine v3.0

Initializes:
1. SQLite database and schema
2. Scoring engine with default parameters
3. Validates environment
"""

import sys
import os
from datetime import datetime
import pytz

# Add SRC to path (init_v3.py now lives in scripts/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SRC'))

from SRC.common import DatabaseSchema, ScoringEngine

ist = pytz.timezone('Asia/Kolkata')


def print_header():
    """Print welcome header"""
    print("\n" + "█"*70)
    print("█ STOCK PREDICTION ENGINE v3.0 - INITIALIZATION")
    print("█ Continuous Scoring System Setup")
    print("█"*70 + "\n")


def check_dependencies():
    """Check if all required packages are available"""
    print("Checking dependencies...")
    
    required_packages = [
        'pandas',
        'numpy',
        'torch',
        'transformers',
        'yfinance',
        'GoogleNews',
        'scipy',
        'pytz',
        'sklearn',
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (MISSING)")
            missing.append(package)
    
    if missing:
        print(f"\n✗ Missing packages: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies available\n")
    return True


def initialize_database():
    """Initialize SQLite database"""
    print("Initializing database...")
    
    db_schema = DatabaseSchema()
    success = db_schema.init_schema()
    
    if success:
        print("\n✓ Database initialized successfully\n")
        db_schema.get_schema_info()
        return db_schema
    else:
        print("\n✗ Database initialization failed\n")
        return None


def initialize_scoring_engine(db_schema):
    """Initialize scoring engine"""
    print("Initializing Scoring Engine...")
    
    engine = ScoringEngine(
        db_path="stock_engine.db",
        normalization_alpha=1.5,
        threshold_bullish=0.1,
        threshold_bearish=-0.1,
        calibration_k=5.0
    )
    
    print("\nDefault Parameters:")
    print(f"  • Normalization (tanh alpha): {engine.normalization_alpha}")
    print(f"  • Bullish threshold: {engine.threshold_bullish}")
    print(f"  • Bearish threshold: {engine.threshold_bearish}")
    print(f"  • Calibration k (initial): {engine.calibration_k}")
    print(f"  • Database: {engine.db_path}")
    
    print("\n✓ Scoring Engine ready\n")
    return engine


def run_demo():
    """Run demo to verify functionality"""
    print("="*70)
    print("RUNNING DEMO")
    print("="*70)
    
    try:
        import pandas as pd
        
        engine = ScoringEngine()
        
        # Demo data
        print("\n1. Creating demo articles...")
        demo_articles = pd.DataFrame({
            'sentiment_score': [0.3, 0.5, -0.1, 0.4, 0.2],
            'is_sponsored': [0, 0, 1, 0, 1]
        })
        print(demo_articles.to_string())
        
        # Aggregate
        print("\n2. Aggregating to stock-level signal...")
        raw_signal = engine.aggregate_articles_to_signal(
            demo_articles,
            'TCS',
            '2024-04-15'
        )
        print(f"   All sentiment raw: {raw_signal['all_sentiment_raw']:.3f}")
        print(f"   Organic sentiment raw: {raw_signal['organic_sentiment_raw']:.3f}")
        print(f"   Sponsored sentiment raw: {raw_signal['sponsored_sentiment_raw']:.3f}")
        
        # Score
        print("\n3. Computing normalized scores...")
        scores = engine.compute_scores(raw_signal)
        print(f"   Prediction score: {scores['pred_score']:.3f}")
        print(f"   Direction: {scores['pred_direction']}")
        print(f"   Expected % change: {scores['expected_pct_change']:.2f}%")
        print(f"   Organic score: {scores['organic_pred_score']:.3f}")
        print(f"   Sponsored score: {scores['sponsored_pred_score']:.3f}")
        
        print("\n✓ Demo completed successfully\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def print_next_steps():
    """Print next steps"""
    print("="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
1. STEP 2: Refactor data pipeline
   - Enhance Sentiment.py to detect sponsored articles
   - Modify Scanner.py to save articles to database
   - Update data_loader2.py to store prices in database
   
2. STEP 3: Create scoring pipeline
   - Build scoring_pipeline.py to aggregate and score daily
   
3. STEP 4: Evaluation & Reporting
   - Create evaluation_metrics.py
   - Create calibration_learner.py
   - Create reporting.py
   
4. STEP 5: Update app.py orchestrator
   - Add new CLI commands (calibrate, score, evaluate, report)
   - Maintain backward compatibility
   - Add scheduled runs with database idempotency

5. STEP 6: Calibration & Backtesting
   - Build backtester.py
   - Generate calibration reports

Next, run:
  python app.py run              # Full pipeline with database persistence

Database location: stock_engine.db
Architecture doc: ARCHITECTURE_v3.md
    """)


def main():
    """Main initialization flow"""
    print_header()
    
    # 1. Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # 2. Initialize database
    db_schema = initialize_database()
    if not db_schema:
        sys.exit(1)
    
    # 3. Initialize scoring engine
    engine = initialize_scoring_engine(db_schema)
    
    # 4. Run demo
    if run_demo():
        print("✓ Initialization successful! System is ready.\n")
    else:
        print("⚠ Initialization completed with demo warnings.\n")
    
    # 5. Print next steps
    print_next_steps()
    
    print("█"*70 + "\n")


if __name__ == "__main__":
    main()
