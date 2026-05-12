#!/usr/bin/env python3
"""
Quick Demo: ML Training System (Fast Version)

This demo trains on just 2 stocks for 1-2 years to show the system working quickly.
For production, use train_ml_model.py with full 10-year dataset.
"""

import sys
import os
from datetime import datetime
import pytz

# Add SRC to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SRC'))

from SRC.past_data import HistoricalBacktester, HistoricalDataGenerator, SponsoredNewsPenaltyLearner

ist = pytz.timezone('Asia/Kolkata')


def main():
    """Run quick demo"""
    print("\n" + "█"*80)
    print("█ ML TRAINING SYSTEM - QUICK DEMO (2 STOCKS, 2 YEARS)")
    print("█ Estimated Runtime: 3-5 minutes")
    print("█"*80 + "\n")
    
    # Use only 2 stocks for speed
    demo_stocks = ['TCS', 'INFY']
    
    # STEP 1: Generate smaller dataset
    print("STEP 1: Generate 2-year historical data...")
    print("-" * 80)
    
    try:
        generator = HistoricalDataGenerator(
            nifty_50_stocks=demo_stocks,
            start_date='2022-04-15',
            end_date='2024-04-15',
            db_path='stock_engine_demo.db'
        )
        
        prices_df = generator.generate_price_data()
        articles_df = generator.generate_news_data(prices_df)
        generator.save_to_database(prices_df, articles_df)
        generator.get_statistics()
        
    except Exception as e:
        print(f"Error in data generation: {e}")
        return False
    
    # STEP 2: Train model
    print("\n\nSTEP 2: Train penalty model...")
    print("-" * 80)
    
    try:
        learner = SponsoredNewsPenaltyLearner(db_path='stock_engine_demo.db')
        
        df = learner.load_training_data()
        if df is None or len(df) < 10:
            print("Insufficient data for training")
            return False
        
        print(f"\nLoaded {len(df):,} training samples\n")
        
        analysis = learner.analyze_sponsored_bias(df)
        penalty_results = learner.learn_penalty_factor(df)
        cal_results = learner.learn_calibration_parameters(df, penalty_results['optimal_penalty'])
        
        learner.generate_report(analysis, penalty_results, cal_results)
        
        print("\n✓ Model training complete\n")
        
        params = {
            'penalty': penalty_results['optimal_penalty'],
            'k': cal_results['k_optimal'],
        }
        
    except Exception as e:
        print(f"Error in model training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # STEP 3: Backtest
    print("\nSTEP 3: Backtest baseline vs optimized...")
    print("-" * 80)
    
    try:
        # Baseline
        print("\nBaseline (no penalty)...")
        bt1 = HistoricalBacktester(db_path='stock_engine_demo.db', sponsored_penalty=0.0, calibration_k=5.0)
        pred1 = bt1.generate_daily_predictions()
        metrics1 = bt1.calculate_metrics(pred1)
        
        # Optimized
        print("Optimized (with penalty)...")
        bt2 = HistoricalBacktester(
            db_path='stock_engine_demo.db',
            sponsored_penalty=params['penalty'],
            calibration_k=params['k']
        )
        pred2 = bt2.generate_daily_predictions()
        metrics2 = bt2.calculate_metrics(pred2)
        
        # Summary
        print("\n\n" + "="*80)
        print("QUICK DEMO RESULTS")
        print("="*80)
        
        print("\nBASELINE (No Penalty):")
        print(f"  Accuracy: {metrics1['accuracy']:.1f}%")
        print(f"  Pearson Correlation: {metrics1['pearson_corr']:.4f}")
        print(f"  RMSE: {metrics1['rmse']:.3f}%")
        
        print(f"\nOPTIMIZED (Penalty={params['penalty']:.3f}, k={params['k']:.3f}):")
        print(f"  Accuracy: {metrics2['accuracy']:.1f}%")
        print(f"  Pearson Correlation: {metrics2['pearson_corr']:.4f}")
        print(f"  RMSE: {metrics2['rmse']:.3f}%")
        
        print("\nIMPROVEMENT:")
        print(f"  Accuracy: {metrics2['accuracy'] - metrics1['accuracy']:+.1f}%")
        print(f"  Correlation: {metrics2['pearson_corr'] - metrics1['pearson_corr']:+.4f}")
        print(f"  RMSE: {metrics1['rmse'] - metrics2['rmse']:+.3f}% ({100*(metrics1['rmse']-metrics2['rmse'])/metrics1['rmse']:+.1f}%)")
        
        print("\n" + "="*80)
        print("✓ DEMO COMPLETE")
        print("="*80)
        
        print("""
For production:
  1. Run: python train_ml_model.py (full 10-year training)
  2. Check: ml_trained_config.py (learned parameters)
  3. Deploy: Apply SPONSORED_NEWS_PENALTY and CALIBRATION_K to production
  4. Monitor: Track live prediction accuracy
  5. Retrain: Monthly with new data
        """)
        
    except Exception as e:
        print(f"Error in backtesting: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
