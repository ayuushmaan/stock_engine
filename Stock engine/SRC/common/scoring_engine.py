"""
Scoring Engine for Continuous Stock Prediction

Converts article-level sentiment into stock-level continuous prediction scores
in the range [-1, 1] using robust normalization techniques.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import pytz
from scipy import stats
from typing import Dict, List, Tuple, Optional
import sqlite3

from SRC.common.paths import DB_DIR, ensure_dirs


class ScoringEngine:
    """
    Core scoring logic for converting sentiment → continuous prediction scores
    
    Scoring Pipeline:
    1. Aggregate article-level sentiment to stock-level raw signal
    2. Normalize raw signal to [-1, 1] using tanh (robust to outliers)
    3. Classify direction from score thresholds
    4. Optionally map to expected % change using calibration factor k
    """
    
    def __init__(self, 
                 db_path: str = str(DB_DIR / "stock_engine.db"),
                 normalization_alpha: float = 1.5,
                 threshold_bullish: float = 0.1,
                 threshold_bearish: float = -0.1,
                 calibration_k: float = 5.0):
        """
        Initialize Scoring Engine
        
        Args:
            db_path: Path to SQLite database
            normalization_alpha: Parameter for tanh normalization (higher = more aggressive)
            threshold_bullish: Score threshold for BULLISH classification (default +0.1)
            threshold_bearish: Score threshold for BEARISH classification (default -0.1)
            calibration_k: Initial scaling factor for % move conversion
        """
        self.db_path = db_path
        self.normalization_alpha = normalization_alpha
        self.threshold_bullish = threshold_bullish
        self.threshold_bearish = threshold_bearish
        self.calibration_k = calibration_k
        self.ist = pytz.timezone('Asia/Kolkata')
    
    # ==================== NORMALIZATION FUNCTIONS ====================
    
    @staticmethod
    def tanh_normalize(raw_signal: float, alpha: float = 1.5) -> float:
        """
        Normalize raw sentiment signal to [-1, 1] using tanh
        
        Benefits:
        - Robust to outliers (tanh squashes extreme values)
        - Non-linear (more sensitivity near zero)
        - Bounded to (-1, 1)
        
        Args:
            raw_signal: Raw mean sentiment (approx. -1 to 1)
            alpha: Scaling parameter (typical range: 1.0-2.0)
            
        Returns:
            Normalized score in [-1, 1]
        """
        score = np.tanh(alpha * raw_signal)
        return np.clip(score, -1.0, 1.0)
    
    @staticmethod
    def zscore_normalize(raw_signals: np.ndarray, n_std: float = 3.0) -> np.ndarray:
        """
        Alternative normalization: Clipped z-score (for Gaussian data)
        
        Args:
            raw_signals: Array of raw sentiment values
            n_std: Number of standard deviations to clip at (default 3)
            
        Returns:
            Normalized scores clipped to [-1, 1]
        """
        mu = np.mean(raw_signals)
        sigma = np.std(raw_signals)
        
        if sigma == 0:
            return np.zeros_like(raw_signals)
        
        z_scores = (raw_signals - mu) / sigma
        normalized = np.clip(z_scores / n_std, -1.0, 1.0)
        return normalized
    
    # ==================== AGGREGATION ====================
    
    def aggregate_articles_to_signal(self, 
                                    articles: pd.DataFrame,
                                    stock_symbol: str,
                                    collection_date: str) -> Dict:
        """
        Aggregate article-level sentiment to stock-level signal
        
        Args:
            articles: DataFrame with columns:
                - sentiment_score: article-level sentiment
                - is_sponsored: 1 if sponsored, 0 if organic
            stock_symbol: Stock ticker
            collection_date: Date of collection (YYYY-MM-DD)
            
        Returns:
            Dictionary with aggregated metrics:
            {
                'stock_symbol': str,
                'collection_date': str,
                'article_count': int,
                'sponsored_count': int,
                'organic_count': int,
                'all_sentiment_raw': float,        # mean(all)
                'sponsored_sentiment_raw': float,  # mean(sponsored)
                'organic_sentiment_raw': float,    # mean(organic)
            }
        """
        result = {
            'stock_symbol': stock_symbol,
            'collection_date': collection_date,
            'article_count': len(articles),
            'sponsored_count': 0,
            'organic_count': 0,
            'all_sentiment_raw': None,
            'sponsored_sentiment_raw': None,
            'organic_sentiment_raw': None,
        }
        
        if len(articles) == 0:
            return result
        
        # All articles
        result['all_sentiment_raw'] = articles['sentiment_score'].mean()
        
        # Sponsored articles
        sponsored = articles[articles['is_sponsored'] == 1]
        if len(sponsored) > 0:
            result['sponsored_count'] = len(sponsored)
            result['sponsored_sentiment_raw'] = sponsored['sentiment_score'].mean()
        
        # Organic (non-sponsored) articles
        organic = articles[articles['is_sponsored'] == 0]
        if len(organic) > 0:
            result['organic_count'] = len(organic)
            result['organic_sentiment_raw'] = organic['sentiment_score'].mean()
        
        return result
    
    # ==================== SCORING ====================
    
    def compute_scores(self, 
                      raw_signal_dict: Dict,
                      use_calibration: bool = True) -> Dict:
        """
        Convert aggregated raw signals to normalized prediction scores
        
        Args:
            raw_signal_dict: Output from aggregate_articles_to_signal()
            use_calibration: If True, compute expected_pct_change using calibration_k
            
        Returns:
            Dictionary with prediction scores and direction:
            {
                'pred_score': float in [-1, 1],
                'sponsored_pred_score': float or None,
                'organic_pred_score': float or None,
                'pred_direction': str ('BULLISH', 'NEUTRAL', 'BEARISH'),
                'expected_pct_change': float or None,
                'normalization_alpha': float,
            }
        """
        result = {
            'pred_score': None,
            'sponsored_pred_score': None,
            'organic_pred_score': None,
            'pred_direction': 'SKIP',
            'expected_pct_change': None,
            'normalization_alpha': self.normalization_alpha,
        }
        
        # All articles
        if raw_signal_dict['all_sentiment_raw'] is not None:
            raw = raw_signal_dict['all_sentiment_raw']
            result['pred_score'] = self.tanh_normalize(raw, self.normalization_alpha)
            
            # Direction classification
            if result['pred_score'] > self.threshold_bullish:
                result['pred_direction'] = 'BULLISH'
            elif result['pred_score'] < self.threshold_bearish:
                result['pred_direction'] = 'BEARISH'
            else:
                result['pred_direction'] = 'NEUTRAL'
            
            # Expected % change
            if use_calibration:
                result['expected_pct_change'] = result['pred_score'] * self.calibration_k
        
        # Sponsored subset
        if raw_signal_dict['sponsored_sentiment_raw'] is not None:
            raw = raw_signal_dict['sponsored_sentiment_raw']
            result['sponsored_pred_score'] = self.tanh_normalize(raw, self.normalization_alpha)
        
        # Organic subset
        if raw_signal_dict['organic_sentiment_raw'] is not None:
            raw = raw_signal_dict['organic_sentiment_raw']
            result['organic_pred_score'] = self.tanh_normalize(raw, self.normalization_alpha)
        
        return result
    
    # ==================== DATABASE OPERATIONS ====================
    
    def save_daily_signals(self, 
                          signal_data: Dict,
                          score_data: Dict,
                          is_production: int = 1) -> bool:
        """
        Save aggregated signals and scores to database
        
        Args:
            signal_data: From aggregate_articles_to_signal()
            score_data: From compute_scores()
            is_production: 1 for production, 0 for test
            
        Returns:
            Boolean success indicator
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO daily_signals (
                    collection_date,
                    stock_symbol,
                    article_count,
                    sponsored_count,
                    organic_count,
                    
                    all_sentiment_raw,
                    sponsored_sentiment_raw,
                    organic_sentiment_raw,
                    
                    pred_score,
                    sponsored_pred_score,
                    organic_pred_score,
                    
                    pred_direction,
                    direction_threshold,
                    expected_pct_change,
                    calibration_k,
                    
                    normalization_alpha,
                    is_production,
                    processing_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_data['collection_date'],
                signal_data['stock_symbol'],
                signal_data['article_count'],
                signal_data['sponsored_count'],
                signal_data['organic_count'],
                signal_data['all_sentiment_raw'],
                signal_data['sponsored_sentiment_raw'],
                signal_data['organic_sentiment_raw'],
                score_data['pred_score'],
                score_data['sponsored_pred_score'],
                score_data['organic_pred_score'],
                score_data['pred_direction'],
                self.threshold_bullish,
                score_data['expected_pct_change'],
                self.calibration_k,
                score_data['normalization_alpha'],
                is_production,
                datetime.now(self.ist).isoformat()
            ))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving signals: {e}")
            return False
        finally:
            conn.close()
    
    # ==================== CALIBRATION ====================
    
    def calibrate_k_factor(self, 
                          lookback_days: int = 90,
                          test_fraction: float = 0.2) -> Dict:
        """
        Learn optimal calibration factor k from historical data
        
        Finds k that minimizes: Σ(k * pred_score_i - actual_pct_change_i)²
        
        Args:
            lookback_days: Historical period to use for calibration
            test_fraction: Fraction of data to reserve for testing
            
        Returns:
            Calibration results:
            {
                'k_optimal': float,
                'rmse_train': float,
                'rmse_test': float,
                'pearson_corr_train': float,
                'pearson_corr_test': float,
                'sample_count_train': int,
                'sample_count_test': int,
                'success': bool,
            }
        """
        result = {
            'k_optimal': self.calibration_k,  # Default fallback
            'rmse_train': None,
            'rmse_test': None,
            'pearson_corr_train': None,
            'pearson_corr_test': None,
            'sample_count_train': 0,
            'sample_count_test': 0,
            'success': False,
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Fetch merged signals + prices
            query = f'''
                SELECT 
                    ds.pred_score,
                    dp.actual_pct_change
                FROM daily_signals ds
                JOIN daily_prices dp 
                    ON ds.collection_date = dp.date 
                    AND ds.stock_symbol = dp.stock_symbol
                WHERE ds.pred_score IS NOT NULL
                    AND dp.actual_pct_change IS NOT NULL
                    AND ds.is_production = 1
                    AND ds.collection_date >= date('now', '-{lookback_days} days')
                ORDER BY ds.collection_date
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(df) < 10:
                print(f"⚠ Insufficient data for calibration (n={len(df)}, need >10)")
                return result
            
            # Train/test split
            n = len(df)
            n_train = int(n * (1 - test_fraction))
            
            df_train = df.iloc[:n_train]
            df_test = df.iloc[n_train:]
            
            # Grid search for optimal k
            k_candidates = np.linspace(1, 15, 100)
            best_rmse = np.inf
            best_k = self.calibration_k
            
            for k in k_candidates:
                pred_pct_train = k * df_train['pred_score'].values
                actual_pct_train = df_train['actual_pct_change'].values
                
                rmse = np.sqrt(np.mean((pred_pct_train - actual_pct_train) ** 2))
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_k = k
            
            result['k_optimal'] = best_k
            
            # Evaluate on train and test
            pred_pct_train = best_k * df_train['pred_score'].values
            actual_pct_train = df_train['actual_pct_change'].values
            result['rmse_train'] = np.sqrt(np.mean((pred_pct_train - actual_pct_train) ** 2))
            result['pearson_corr_train'] = np.corrcoef(
                df_train['pred_score'].values,
                df_train['actual_pct_change'].values
            )[0, 1]
            result['sample_count_train'] = len(df_train)
            
            if len(df_test) > 0:
                pred_pct_test = best_k * df_test['pred_score'].values
                actual_pct_test = df_test['actual_pct_change'].values
                result['rmse_test'] = np.sqrt(np.mean((pred_pct_test - actual_pct_test) ** 2))
                result['pearson_corr_test'] = np.corrcoef(
                    df_test['pred_score'].values,
                    df_test['actual_pct_change'].values
                )[0, 1]
                result['sample_count_test'] = len(df_test)
            
            result['success'] = True
            return result
            
        except Exception as e:
            print(f"Error during calibration: {e}")
            return result
    
    def save_calibration(self, 
                        calibration_date: str,
                        lookback_days: int,
                        calibration_results: Dict) -> bool:
        """
        Save calibration results to database for future use
        
        Args:
            calibration_date: Date of calibration (YYYY-MM-DD)
            lookback_days: Training period used
            calibration_results: From calibrate_k_factor()
            
        Returns:
            Boolean success indicator
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Deactivate previous calibrations
            cursor.execute("UPDATE calibration_history SET is_active = 0")
            
            # Insert new calibration
            cursor.execute('''
                INSERT INTO calibration_history (
                    calibration_date,
                    lookback_days,
                    k_optimal,
                    threshold_bullish,
                    threshold_bearish,
                    normalization_param,
                    rmse_on_train,
                    rmse_on_test,
                    pearson_on_train,
                    pearson_on_test,
                    is_active,
                    train_sample_count,
                    test_sample_count,
                    created_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                calibration_date,
                lookback_days,
                calibration_results['k_optimal'],
                self.threshold_bullish,
                self.threshold_bearish,
                self.normalization_alpha,
                calibration_results['rmse_train'],
                calibration_results['rmse_test'],
                calibration_results['pearson_corr_train'],
                calibration_results['pearson_corr_test'],
                1,  # is_active
                calibration_results['sample_count_train'],
                calibration_results['sample_count_test'],
                datetime.now(self.ist).isoformat()
            ))
            
            conn.commit()
            
            # Update engine parameters
            self.calibration_k = calibration_results['k_optimal']
            
            print(f"✓ Calibration saved (k={calibration_results['k_optimal']:.3f})")
            return True
            
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False
        finally:
            conn.close()
    
    def load_active_calibration(self) -> bool:
        """
        Load active calibration parameters from database
        
        Returns:
            Boolean success indicator
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT k_optimal, threshold_bullish, threshold_bearish, normalization_param
                FROM calibration_history
                WHERE is_active = 1
                ORDER BY calibration_date DESC
                LIMIT 1
            ''')
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                self.calibration_k, self.threshold_bullish, self.threshold_bearish, self.normalization_alpha = result
                print(f"✓ Loaded calibration: k={self.calibration_k:.3f}, alpha={self.normalization_alpha:.3f}")
                return True
            else:
                print("⚠ No active calibration found, using defaults")
                return False
                
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False


if __name__ == "__main__":
    # Demo
    from SRC.common.db_schema import DatabaseSchema
    
    # Initialize DB
    db_schema = DatabaseSchema()
    db_schema.init_schema()
    
    # Initialize scoring engine
    engine = ScoringEngine()
    
    # Demo aggregation
    demo_articles = pd.DataFrame({
        'sentiment_score': [0.3, 0.5, -0.1, 0.4, 0.2],
        'is_sponsored': [0, 0, 1, 0, 1]
    })
    
    print("\n" + "="*70)
    print("SCORING ENGINE DEMO")
    print("="*70)
    print("\nInput articles:")
    print(demo_articles)
    
    # Aggregate
    raw_signal = engine.aggregate_articles_to_signal(
        demo_articles,
        'TCS',
        '2024-04-15'
    )
    print("\nAggregated raw signal:")
    for k, v in raw_signal.items():
        print(f"  {k}: {v}")
    
    # Score
    scores = engine.compute_scores(raw_signal)
    print("\nNormalized scores:")
    for k, v in scores.items():
        print(f"  {k}: {v}")
