"""
Historical Backtester - Validate Model Performance

Compares:
1. Predictions WITHOUT sponsored penalty (baseline)
2. Predictions WITH learned sponsored penalty (optimized)

Generates comprehensive backtest reports with:
- Daily prediction accuracy
- Monthly performance
- Cumulative P&L
- Risk metrics
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import pytz

class HistoricalBacktester:
    """Backtest predictions against historical prices"""
    
    def __init__(self, 
                 db_path='stock_engine_historical.db',
                 sponsored_penalty=0.0,
                 calibration_k=5.0):
        """
        Initialize backtester
        
        Args:
            db_path: Historical data database path
            sponsored_penalty: Penalty factor for sponsored news
            calibration_k: Scaling factor for % predictions
        """
        self.db_path = db_path
        self.sponsored_penalty = sponsored_penalty
        self.calibration_k = calibration_k
        self.ist = pytz.timezone('Asia/Kolkata')
    
    def generate_daily_predictions(self):
        """Generate daily predictions from aggregated article sentiment"""
        print(f"🎯 Generating daily predictions (penalty={self.sponsored_penalty:.3f})...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Get daily aggregated sentiment and prices
        query = '''
            SELECT 
                a.collection_date as date,
                a.stock_symbol,
                COUNT(*) as article_count,
                SUM(CASE WHEN a.is_sponsored = 0 THEN 1 ELSE 0 END) as organic_count,
                SUM(CASE WHEN a.is_sponsored = 1 THEN 1 ELSE 0 END) as sponsored_count,
                AVG(a.sentiment_score) as all_sentiment_raw,
                AVG(CASE WHEN a.is_sponsored = 0 THEN a.sentiment_score END) as organic_sentiment_raw,
                AVG(CASE WHEN a.is_sponsored = 1 THEN a.sentiment_score END) as sponsored_sentiment_raw,
                p.actual_pct_change,
                p.actual_direction
            FROM articles a
            LEFT JOIN daily_prices p ON a.collection_date = p.date AND a.stock_symbol = p.stock_symbol
            WHERE a.sentiment_score IS NOT NULL
            GROUP BY a.collection_date, a.stock_symbol
            ORDER BY a.collection_date, a.stock_symbol
        '''
        
        raw_data = pd.read_sql_query(query, conn)
        conn.close()
        
        # Apply penalty and generate predictions
        predictions = []
        
        for _, row in raw_data.iterrows():
            date = row['date']
            symbol = row['stock_symbol']
            all_sentiment = row['all_sentiment_raw']
            organic_sentiment = row['organic_sentiment_raw']
            sponsored_sentiment = row['sponsored_sentiment_raw']
            
            # Calculate weighted sentiment with penalty
            total_articles = row['article_count']
            organic_count = row['organic_count'] or 0
            sponsored_count = row['sponsored_count'] or 0
            
            # Apply penalty to sponsored sentiment
            if sponsored_count > 0 and sponsored_sentiment:
                penalized_sponsored = sponsored_sentiment * (1 - self.sponsored_penalty)
            else:
                penalized_sponsored = sponsored_sentiment or 0
            
            # Weighted average: (organic_count * organic + sponsored_count * penalized_sponsored) / total
            if total_articles > 0:
                adjusted_sentiment = (
                    (organic_count * (organic_sentiment or 0) + 
                     sponsored_count * penalized_sponsored) / total_articles
                ) if total_articles > 0 else 0
            else:
                adjusted_sentiment = 0
            
            # Generate prediction
            pred_score = np.tanh(1.5 * adjusted_sentiment)  # Normalize
            expected_pct = self.calibration_k * pred_score
            
            if pred_score > 0.1:
                pred_direction = 'BULLISH'
            elif pred_score < -0.1:
                pred_direction = 'BEARISH'
            else:
                pred_direction = 'NEUTRAL'
            
            # Evaluate prediction
            actual_pct = row['actual_pct_change'] or 0
            actual_direction = row['actual_direction'] or 'NEUTRAL'
            
            correct = (
                (pred_direction == 'BULLISH' and actual_pct > 0.5) or
                (pred_direction == 'BEARISH' and actual_pct < -0.5) or
                (pred_direction == 'NEUTRAL' and -0.5 <= actual_pct <= 0.5)
            )
            
            predictions.append({
                'date': date,
                'symbol': symbol,
                'article_count': total_articles,
                'organic_count': organic_count,
                'sponsored_count': sponsored_count,
                'all_sentiment_raw': all_sentiment,
                'organic_sentiment_raw': organic_sentiment,
                'sponsored_sentiment_raw': sponsored_sentiment,
                'adjusted_sentiment': adjusted_sentiment,
                'pred_score': pred_score,
                'pred_direction': pred_direction,
                'expected_pct': expected_pct,
                'actual_pct': actual_pct,
                'actual_direction': actual_direction,
                'correct': 1 if correct else 0,
                'error_pct': abs(expected_pct - actual_pct),
            })
        
        df = pd.DataFrame(predictions)
        print(f"✓ Generated {len(df):,} daily predictions")
        
        return df
    
    def calculate_metrics(self, predictions_df):
        """Calculate performance metrics"""
        print(f"\n📊 Calculating metrics...")
        
        # Overall metrics
        total_preds = len(predictions_df)
        correct_preds = predictions_df['correct'].sum()
        accuracy = (correct_preds / total_preds * 100) if total_preds > 0 else 0
        
        # Correlation metrics
        pearson_corr = predictions_df[['pred_score', 'actual_pct']].corr().iloc[0, 1]
        spearman_corr = predictions_df[['pred_score', 'actual_pct']].corr(method='spearman').iloc[0, 1]
        
        # Error metrics
        mae = predictions_df['error_pct'].mean()
        rmse = np.sqrt((predictions_df['error_pct'] ** 2).mean())
        mape = ((predictions_df['error_pct'] / (100 * np.abs(predictions_df['actual_pct']) + 0.01)).mean()) * 100
        
        # Directional accuracy
        bullish_preds = predictions_df[predictions_df['pred_direction'] == 'BULLISH']
        bullish_correct = ((bullish_preds['actual_pct'] > 0.5).sum() / len(bullish_preds) * 100) if len(bullish_preds) > 0 else 0
        
        bearish_preds = predictions_df[predictions_df['pred_direction'] == 'BEARISH']
        bearish_correct = ((bearish_preds['actual_pct'] < -0.5).sum() / len(bearish_preds) * 100) if len(bearish_preds) > 0 else 0
        
        # Separate metrics by news type
        organic_preds = predictions_df[predictions_df['organic_count'] > 0]
        organic_corr = organic_preds[['pred_score', 'actual_pct']].corr().iloc[0, 1] if len(organic_preds) > 0 else 0
        organic_acc = (organic_preds['correct'].sum() / len(organic_preds) * 100) if len(organic_preds) > 0 else 0
        
        sponsored_preds = predictions_df[(predictions_df['sponsored_count'] > 0) & (predictions_df['organic_count'] == 0)]
        sponsored_corr = sponsored_preds[['pred_score', 'actual_pct']].corr().iloc[0, 1] if len(sponsored_preds) > 0 else 0
        sponsored_acc = (sponsored_preds['correct'].sum() / len(sponsored_preds) * 100) if len(sponsored_preds) > 0 else 0
        
        metrics = {
            'total_predictions': total_preds,
            'correct_predictions': correct_preds,
            'accuracy': accuracy,
            'pearson_corr': pearson_corr,
            'spearman_corr': spearman_corr,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'bullish_accuracy': bullish_correct,
            'bearish_accuracy': bearish_correct,
            'organic_corr': organic_corr,
            'organic_accuracy': organic_acc,
            'organic_count': len(organic_preds),
            'sponsored_only_corr': sponsored_corr,
            'sponsored_only_accuracy': sponsored_acc,
            'sponsored_only_count': len(sponsored_preds),
        }
        
        return metrics
    
    def monthly_performance(self, predictions_df):
        """Calculate monthly aggregated performance"""
        
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        predictions_df['year_month'] = predictions_df['date'].dt.to_period('M')
        
        monthly = predictions_df.groupby('year_month').agg({
            'correct': ['sum', 'count'],
            'error_pct': 'mean',
            'actual_pct': ['mean', 'std'],
            'pred_score': 'mean'
        }).round(3)
        
        monthly.columns = ['Correct', 'Total', 'MAE', 'Avg_Actual_Pct', 'Volatility', 'Avg_Pred_Score']
        monthly['Accuracy'] = (monthly['Correct'] / monthly['Total'] * 100).round(1)
        
        return monthly.drop(['Correct', 'Total'], axis=1)
    
    def generate_backtest_report(self, metrics, monthly_perf, predictions_df, model_name=""):
        """Generate comprehensive backtest report"""
        
        print("\n" + "="*80)
        if model_name:
            print(f"BACKTEST REPORT: {model_name}")
        else:
            print("BACKTEST REPORT")
        print("="*80)
        
        print(f"\n📈 Overall Performance")
        print("-" * 80)
        print(f"Total predictions:          {metrics['total_predictions']:,}")
        print(f"Correct predictions:        {metrics['correct_predictions']:,}")
        print(f"Accuracy:                   {metrics['accuracy']:.1f}%")
        print(f"Pearson correlation:        {metrics['pearson_corr']:.4f}")
        print(f"Spearman correlation:       {metrics['spearman_corr']:.4f}")
        
        print(f"\n📊 Error Metrics")
        print("-" * 80)
        print(f"MAE (Mean Absolute Error):  {metrics['mae']:.3f}%")
        print(f"RMSE:                       {metrics['rmse']:.3f}%")
        print(f"MAPE:                       {metrics['mape']:.1f}%")
        
        print(f"\n🎯 Directional Accuracy")
        print("-" * 80)
        print(f"Bullish prediction accuracy: {metrics['bullish_accuracy']:.1f}%")
        print(f"Bearish prediction accuracy: {metrics['bearish_accuracy']:.1f}%")
        
        print(f"\n🔍 Organic vs Sponsored News")
        print("-" * 80)
        print(f"Organic news (n={metrics['organic_count']:,}):")
        print(f"  Correlation: {metrics['organic_corr']:.4f}")
        print(f"  Accuracy:    {metrics['organic_accuracy']:.1f}%")
        print(f"\nSponsored-only (n={metrics['sponsored_only_count']:,}):")
        print(f"  Correlation: {metrics['sponsored_only_corr']:.4f}")
        print(f"  Accuracy:    {metrics['sponsored_only_accuracy']:.1f}%")
        
        if metrics['organic_corr'] > 0 and metrics['sponsored_only_corr'] > 0:
            improvement = ((metrics['organic_corr'] - metrics['sponsored_only_corr']) / 
                          metrics['sponsored_only_corr'] * 100)
            print(f"\n  → Organic news is {improvement:.0f}% more predictive")
        
        print(f"\n📅 Monthly Performance (Sample)")
        print("-" * 80)
        print(monthly_perf.head(12).to_string())
        
        print("\n" + "="*80)


def main():
    """Run backtest"""
    print("\n" + "█"*80)
    print("█ HISTORICAL BACKTESTER - MODEL VALIDATION")
    print("█"*80)
    
    # Test 1: Baseline (no penalty)
    print("\n\n--- TEST 1: BASELINE (NO PENALTY) ---\n")
    bt_baseline = HistoricalBacktester(sponsored_penalty=0.0, calibration_k=5.0)
    pred_baseline = bt_baseline.generate_daily_predictions()
    metrics_baseline = bt_baseline.calculate_metrics(pred_baseline)
    monthly_baseline = bt_baseline.monthly_performance(pred_baseline)
    bt_baseline.generate_backtest_report(metrics_baseline, monthly_baseline, pred_baseline, "Baseline (No Penalty)")
    
    # Test 2: With learned penalty
    print("\n\n--- TEST 2: WITH LEARNED PENALTY ---\n")
    bt_optimized = HistoricalBacktester(sponsored_penalty=0.6, calibration_k=5.8)
    pred_optimized = bt_optimized.generate_daily_predictions()
    metrics_optimized = bt_optimized.calculate_metrics(pred_optimized)
    monthly_optimized = bt_optimized.monthly_performance(pred_optimized)
    bt_optimized.generate_backtest_report(metrics_optimized, monthly_optimized, pred_optimized, "Optimized (With Penalty=0.6)")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON: BASELINE vs OPTIMIZED")
    print("="*80)
    
    comparison = {
        'Metric': ['Accuracy', 'Pearson Corr', 'RMSE', 'Bullish Accuracy', 'Organic Corr'],
        'Baseline': [
            f"{metrics_baseline['accuracy']:.1f}%",
            f"{metrics_baseline['pearson_corr']:.4f}",
            f"{metrics_baseline['rmse']:.3f}%",
            f"{metrics_baseline['bullish_accuracy']:.1f}%",
            f"{metrics_baseline['organic_corr']:.4f}",
        ],
        'Optimized': [
            f"{metrics_optimized['accuracy']:.1f}%",
            f"{metrics_optimized['pearson_corr']:.4f}",
            f"{metrics_optimized['rmse']:.3f}%",
            f"{metrics_optimized['bullish_accuracy']:.1f}%",
            f"{metrics_optimized['organic_corr']:.4f}",
        ],
        'Change': [
            f"{metrics_optimized['accuracy'] - metrics_baseline['accuracy']:+.1f}%",
            f"{metrics_optimized['pearson_corr'] - metrics_baseline['pearson_corr']:+.4f}",
            f"{metrics_optimized['rmse'] - metrics_baseline['rmse']:+.3f}%",
            f"{metrics_optimized['bullish_accuracy'] - metrics_baseline['bullish_accuracy']:+.1f}%",
            f"{metrics_optimized['organic_corr'] - metrics_baseline['organic_corr']:+.4f}",
        ]
    }
    
    comp_df = pd.DataFrame(comparison)
    print("\n" + comp_df.to_string(index=False))
    
    print("\n✓ Backtest complete!")


if __name__ == "__main__":
    main()
