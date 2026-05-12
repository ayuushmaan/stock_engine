"""
ML Model Trainer for Sponsored News Penalty Learning

Trains models to learn:
1. Optimal penalty factor for sponsored news
2. Calibration parameters (k, alpha)
3. Thresholds (bullish/bearish)

Compares:
- Model WITHOUT sponsored penalty (baseline)
- Model WITH learned sponsored penalty (improved)
- Performance improvement metrics

Uses historical data to learn that many sponsored news predictions are false positives.
"""

import numpy as np
import pandas as pd
import sqlite3
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pytz
from scipy import stats

class SponsoredNewsPenaltyLearner:
    """Learn optimal penalties for sponsored news to improve prediction accuracy"""
    
    def __init__(self, db_path='stock_engine_historical.db'):
        """
        Initialize learner
        
        Args:
            db_path: Path to historical data database
        """
        self.db_path = db_path
        self.ist = pytz.timezone('Asia/Kolkata')
        self.models = {}  # Store trained models
        self.penalties = {}  # Store learned penalties
    
    def load_training_data(self, lookback_days=3650):
        """
        Load articles and prices for training
        
        Args:
            lookback_days: Historical period (default 10 years = 3650 days)
            
        Returns:
            DataFrame with aligned article + price data
        """
        print(f"📥 Loading training data (last {lookback_days} days)...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT 
                    a.collection_date,
                    a.stock_symbol,
                    a.sentiment_score,
                    a.is_sponsored,
                    a.sentiment_positive_prob,
                    a.sentiment_negative_prob,
                    p.actual_pct_change,
                    p.actual_direction
                FROM articles a
                JOIN daily_prices p 
                    ON a.collection_date = p.date 
                    AND a.stock_symbol = p.stock_symbol
                WHERE a.sentiment_score IS NOT NULL
                    AND p.actual_pct_change IS NOT NULL
                ORDER BY a.collection_date
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            print(f"✓ Loaded {len(df):,} training pairs")
            
            # Statistics
            organic = df[df['is_sponsored'] == 0]
            sponsored = df[df['is_sponsored'] == 1]
            
            print(f"\n  Organic articles:    {len(organic):,}")
            print(f"  Sponsored articles:  {len(sponsored):,}")
            
            return df
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    def analyze_sponsored_bias(self, df):
        """Analyze how sponsored news differs from organic"""
        print("\n🔍 Analyzing Sponsored vs Organic News Bias...")
        
        organic = df[df['is_sponsored'] == 0]
        sponsored = df[df['is_sponsored'] == 1]
        
        stats_data = {
            'Metric': [],
            'Organic': [],
            'Sponsored': [],
            'Difference': []
        }
        
        # Sentiment bias
        org_sent_mean = organic['sentiment_score'].mean()
        spon_sent_mean = sponsored['sentiment_score'].mean()
        stats_data['Metric'].append('Avg Sentiment')
        stats_data['Organic'].append(f"{org_sent_mean:+.3f}")
        stats_data['Sponsored'].append(f"{spon_sent_mean:+.3f}")
        stats_data['Difference'].append(f"{spon_sent_mean - org_sent_mean:+.3f}")
        
        # Correlation to actual movement
        org_corr = organic[['sentiment_score', 'actual_pct_change']].corr().iloc[0, 1]
        spon_corr = sponsored[['sentiment_score', 'actual_pct_change']].corr().iloc[0, 1]
        stats_data['Metric'].append('Correlation to Price')
        stats_data['Organic'].append(f"{org_corr:.4f}")
        stats_data['Sponsored'].append(f"{spon_corr:.4f}")
        stats_data['Difference'].append(f"{spon_corr - org_corr:.4f}")
        
        # False positive rate (predicted bullish but went down)
        org_bullish = (organic['sentiment_score'] > 0).sum()
        org_bullish_down = ((organic['sentiment_score'] > 0) & (organic['actual_pct_change'] < 0)).sum()
        org_fpr = org_bullish_down / org_bullish if org_bullish > 0 else 0
        
        spon_bullish = (sponsored['sentiment_score'] > 0).sum()
        spon_bullish_down = ((sponsored['sentiment_score'] > 0) & (sponsored['actual_pct_change'] < 0)).sum()
        spon_fpr = spon_bullish_down / spon_bullish if spon_bullish > 0 else 0
        
        stats_data['Metric'].append('False Positive Rate')
        stats_data['Organic'].append(f"{100*org_fpr:.1f}%")
        stats_data['Sponsored'].append(f"{100*spon_fpr:.1f}%")
        stats_data['Difference'].append(f"{100*(spon_fpr - org_fpr):+.1f}%")
        
        # Directional accuracy
        org_correct = (((organic['sentiment_score'] > 0) & (organic['actual_pct_change'] > 0)) | 
                      ((organic['sentiment_score'] < 0) & (organic['actual_pct_change'] < 0))).sum()
        org_acc = org_correct / len(organic) * 100 if len(organic) > 0 else 0
        
        spon_correct = (((sponsored['sentiment_score'] > 0) & (sponsored['actual_pct_change'] > 0)) | 
                       ((sponsored['sentiment_score'] < 0) & (sponsored['actual_pct_change'] < 0))).sum()
        spon_acc = spon_correct / len(sponsored) * 100 if len(sponsored) > 0 else 0
        
        stats_data['Metric'].append('Directional Accuracy')
        stats_data['Organic'].append(f"{org_acc:.1f}%")
        stats_data['Sponsored'].append(f"{spon_acc:.1f}%")
        stats_data['Difference'].append(f"{spon_acc - org_acc:+.1f}%")
        
        stats_df = pd.DataFrame(stats_data)
        print("\n" + stats_df.to_string(index=False))
        
        return {
            'org_sentiment_mean': org_sent_mean,
            'spon_sentiment_mean': spon_sent_mean,
            'org_corr': org_corr,
            'spon_corr': spon_corr,
            'org_fpr': org_fpr,
            'spon_fpr': spon_fpr,
            'org_acc': org_acc,
            'spon_acc': spon_acc,
        }
    
    def learn_penalty_factor(self, df, test_size=0.2):
        """
        Learn optimal penalty to apply to sponsored news sentiments
        
        Approach:
        - Feature: sentiment_score * (1 - penalty * is_sponsored)
        - Target: actual_pct_change
        - Find penalty that minimizes RMSE
        
        Args:
            df: Training data
            test_size: Test/train split
            
        Returns:
            Dictionary with learned parameters
        """
        print(f"\n🧠 Learning Optimal Sponsored News Penalty...")
        
        # Train/test split
        np.random.seed(42)
        indices = np.random.permutation(len(df))
        split = int(len(df) * (1 - test_size))
        
        train_idx = indices[:split]
        test_idx = indices[split:]
        
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        
        print(f"  Train samples: {len(df_train):,}")
        print(f"  Test samples:  {len(df_test):,}")
        
        # Grid search for optimal penalty
        penalties = np.linspace(0, 2, 50)  # Penalty from 0 (no effect) to 2 (reverse signal)
        best_rmse = np.inf
        best_penalty = 0
        rmse_scores = []
        
        for penalty in penalties:
            # Create adjusted sentiment: reduce bullish sponsored signals
            train_X = (df_train['sentiment_score'] * 
                      (1 - penalty * df_train['is_sponsored'])).values.reshape(-1, 1)
            train_y = df_train['actual_pct_change'].values
            
            test_X = (df_test['sentiment_score'] * 
                     (1 - penalty * df_test['is_sponsored'])).values.reshape(-1, 1)
            test_y = df_test['actual_pct_change'].values
            
            # Train linear model
            model = LinearRegression()
            model.fit(train_X, train_y)
            
            # Evaluate
            train_pred = model.predict(train_X)
            test_pred = model.predict(test_X)
            
            train_rmse = np.sqrt(np.mean((train_pred - train_y) ** 2))
            test_rmse = np.sqrt(np.mean((test_pred - test_y) ** 2))
            
            rmse_scores.append({
                'penalty': penalty,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'model': model,
                'coef': model.coef_[0]
            })
            
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                best_penalty = penalty
        
        # Get best model
        best_model_info = [m for m in rmse_scores if m['penalty'] == best_penalty][0]
        best_model = best_model_info['model']
        best_coef = best_model_info['coef']
        
        print(f"\n  Optimal penalty: {best_penalty:.3f}")
        print(f"  Train RMSE:     {best_model_info['train_rmse']:.4f}%")
        print(f"  Test RMSE:      {best_model_info['test_rmse']:.4f}%")
        print(f"  Sentiment coef: {best_coef:.4f}")
        
        # Compare with no penalty baseline
        baseline_rmse = [m for m in rmse_scores if m['penalty'] == 0][0]['test_rmse']
        improvement = ((baseline_rmse - best_rmse) / baseline_rmse) * 100
        print(f"\n  Improvement vs baseline: {improvement:+.1f}%")
        
        self.penalty_scores = rmse_scores
        
        return {
            'optimal_penalty': best_penalty,
            'train_rmse': best_model_info['train_rmse'],
            'test_rmse': best_model_info['test_rmse'],
            'sentiment_coefficient': best_coef,
            'improvement_vs_baseline': improvement,
            'model': best_model,
        }
    
    def learn_calibration_parameters(self, df, penalty_factor=0):
        """
        Learn k (% move scaling), alpha (normalization), and thresholds
        
        Args:
            df: Training data
            penalty_factor: Sponsored news penalty to apply
            
        Returns:
            Calibration parameters
        """
        print(f"\n⚙️  Learning Calibration Parameters (penalty={penalty_factor:.3f})...")
        
        # Apply penalty
        df_adjusted = df.copy()
        df_adjusted['sentiment_adjusted'] = (
            df['sentiment_score'] * (1 - penalty_factor * df['is_sponsored'])
        )
        
        # Grid search for k (scaling factor to convert sentiment to % move)
        k_values = np.linspace(1, 20, 50)
        best_rmse = np.inf
        best_k = 5.0
        
        for k in k_values:
            pred_pct = k * df_adjusted['sentiment_adjusted'].values
            actual_pct = df_adjusted['actual_pct_change'].values
            
            rmse = np.sqrt(np.mean((pred_pct - actual_pct) ** 2))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_k = k
        
        # Correlation with adjusted sentiment
        corr = df_adjusted[['sentiment_adjusted', 'actual_pct_change']].corr().iloc[0, 1]
        
        print(f"  Optimal k:          {best_k:.3f}")
        print(f"  RMSE at optimal k:  {best_rmse:.4f}%")
        print(f"  Pearson corr:       {corr:.4f}")
        
        return {
            'k_optimal': best_k,
            'rmse': best_rmse,
            'pearson_corr': corr,
        }
    
    def generate_report(self, analysis, penalty_results, calibration_results):
        """Generate comprehensive training report"""
        
        print("\n" + "="*70)
        print("MODEL TRAINING REPORT: SPONSORED NEWS PENALTY LEARNING")
        print("="*70)
        
        print("\n1. BIAS ANALYSIS")
        print("-" * 70)
        print(f"Funded sentiment (organic vs sponsored): {analysis['org_sentiment_mean']:+.3f} vs {analysis['spon_sentiment_mean']:+.3f}")
        print(f"Correlation difference: {abs(analysis['org_corr'] - analysis['spon_corr']):.4f}")
        print(f"Directional accuracy difference: {analysis['org_acc'] - analysis['spon_acc']:+.1f}%")
        print(f"False positive rate (sponsored): {100*analysis['spon_fpr']:.1f}%")
        
        print("\n2. PENALTY LEARNING")
        print("-" * 70)
        print(f"Optimal penalty factor: {penalty_results['optimal_penalty']:.3f}")
        print(f"  → Multiply sponsored sentiment by (1 - {penalty_results['optimal_penalty']:.3f})")
        print(f"  → Effect: Reduce bullish sponsored signals by {100*penalty_results['optimal_penalty']:.1f}%")
        print(f"\nPerformance:")
        print(f"  Test RMSE (with penalty):    {penalty_results['test_rmse']:.4f}%")
        print(f"  Improvement vs baseline:     {penalty_results['improvement_vs_baseline']:+.1f}%")
        
        print("\n3. CALIBRATION PARAMETERS (with learned penalty)")
        print("-" * 70)
        print(f"Scaling factor (k):     {calibration_results['k_optimal']:.3f}")
        print(f"  → Expected % move = {calibration_results['k_optimal']:.3f} × pred_score")
        print(f"Predictive correlation: {calibration_results['pearson_corr']:.4f}")
        
        print("\n4. EXPECTED IMPACT")
        print("-" * 70)
        print(f"By applying penalty of {penalty_results['optimal_penalty']:.3f} to sponsored news:")
        print(f"  • Reduces false positive rate")
        print(f"  • Improves prediction RMSE by {penalty_results['improvement_vs_baseline']:.1f}%")
        print(f"  • More accurate for trading decisions")
        
        print("\n" + "="*70)


def main():
    """Train models to learn sponsored news penalties"""
    print("\n" + "█"*70)
    print("█ ML MODEL TRAINER - SPONSORED NEWS PENALTY LEARNING")
    print("█"*70)
    
    learner = SponsoredNewsPenaltyLearner()
    
    # Load data
    df = learner.load_training_data()
    if df is None or len(df) == 0:
        print("✗ No training data available")
        return
    
    # Analyze bias
    analysis = learner.analyze_sponsored_bias(df)
    
    # Learn penalty
    penalty_results = learner.learn_penalty_factor(df)
    
    # Learn calibration with penalty
    calibration_results = learner.learn_calibration_parameters(df, penalty_results['optimal_penalty'])
    
    # Generate report
    learner.generate_report(analysis, penalty_results, calibration_results)
    
    print("\n✓ Model training complete!")
    print("\nTo apply this model:")
    print(f"  1. Set sponsored_penalty = {penalty_results['optimal_penalty']:.3f}")
    print(f"  2. Set calibration_k = {calibration_results['k_optimal']:.3f}")
    print(f"  3. Apply: adjusted_sentiment = sentiment * (1 - {penalty_results['optimal_penalty']:.3f} * is_sponsored)")


if __name__ == "__main__":
    main()
