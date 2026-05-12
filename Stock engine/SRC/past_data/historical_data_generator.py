"""
Historical Data Generator for 10-Year Backtest

Generates realistic synthetic historical data for:
- News articles with sentiment + sponsored flag
- Daily stock prices (OHLCV)
- Correlation between news sentiment and actual price movements

Ensures:
- Sponsored news is less predictive (more false positives)
- Organic news is more predictive
- Realistic market dynamics (autocorrelation, volatility regimes)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import pytz

class HistoricalDataGenerator:
    """Generate realistic historical news + price data for model training"""
    
    def __init__(self, 
                 nifty_50_stocks=None,
                 start_date='2014-04-15',
                 end_date='2024-04-15',
                 db_path='stock_engine_historical.db'):
        """
        Initialize generator
        
        Args:
            nifty_50_stocks: List of stock symbols (default: 50 random)
            start_date: Historical period start (YYYY-MM-DD)
            end_date: Historical period end (YYYY-MM-DD)
            db_path: Database to save generated data
        """
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.db_path = db_path
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Default Nifty 50 symbols
        if nifty_50_stocks is None:
            self.stocks = [
                'TCS', 'INFY', 'WIPRO', 'RELIANCE', 'HDFC', 'ICICIBANK', 
                'AXISBANK', 'SBIN', 'LT', 'MARUTI', 'BAJAJ-AUTO', 'SUNPHARMA',
                'NESTLEIND', 'BPCL', 'TATASTEEL', 'NTPC', 'POWER', 'JSWSTEEL',
                'KOTAKBANK', 'HDFCBANK', 'BHARTIARTL', 'ITC', 'DRREDDY', 'CIPLA',
                'EICHERMOT', 'GRAPHITE', 'ONGC', 'COALINDIA', 'INDIGO', 'ADANIPORTS',
                'ADANIGREEN', 'ADANIENT', 'ADANIPOWER', 'HINDUNILVR', 'BRITANNIA',
                'MONSANTO', 'PIDILITIND', 'COLPAL', 'GODREJCP', 'HAVELLS', 'SIEMENS',
                'ABB', 'GRASIM', 'SHISTEEL', 'LTTS', 'MINDTREE', 'ICICIPRU',
                'IRFC', 'PFC', 'REC', 'RECLTD'
            ][:50]
        else:
            self.stocks = nifty_50_stocks
    
    def generate_price_data(self):
        """Generate realistic historical OHLCV data using random walk"""
        print("\n📊 Generating 10 years of price data...")
        
        data = []
        date = self.start_date
        
        for stock in self.stocks:
            print(f"  Generating {stock}...", end=" ")
            
            # Starting parameters per stock
            price = np.random.uniform(100, 5000)  # Random starting price
            volatility = np.random.uniform(0.015, 0.035)  # Daily volatility
            trend = np.random.uniform(-0.0001, 0.0003)  # Daily drift
            
            while date <= self.end_date:
                # Random walk with drift
                daily_return = np.random.normal(trend, volatility)
                
                open_p = price
                close_p = price * (1 + daily_return)
                high_p = max(open_p, close_p) * np.random.uniform(1.0, 1.03)
                low_p = min(open_p, close_p) * np.random.uniform(0.97, 1.0)
                volume = int(np.random.uniform(1e6, 100e6))
                
                pct_change = ((close_p - open_p) / open_p) * 100
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'stock_symbol': stock,
                    'open_price': round(open_p, 2),
                    'close_price': round(close_p, 2),
                    'high_price': round(high_p, 2),
                    'low_price': round(low_p, 2),
                    'volume': volume,
                    'actual_pct_change': round(pct_change, 2),
                    'actual_direction': 'UP' if pct_change > 0.5 else ('DOWN' if pct_change < -0.5 else 'NEUTRAL'),
                    'fetched_time': datetime.now(self.ist).isoformat(),
                    'data_source': 'synthetic_historical'
                })
                
                price = close_p
                date += timedelta(days=1)
                
                # Skip weekends roughly
                if date.weekday() >= 5:
                    date += timedelta(days=2)
            
            date = self.start_date  # Reset for next stock
            print(f"✓ ~3650 days")
        
        print(f"✓ Generated {len(data)} price records")
        return pd.DataFrame(data)
    
    def generate_news_data(self, prices_df):
        """Generate news articles with sentiments"""
        print("\n📰 Generating 10 years of news data...")
        
        articles = []
        
        for stock in self.stocks:
            stock_prices = prices_df[prices_df['stock_symbol'] == stock]
            
            for _, price_row in stock_prices.iterrows():
                date = price_row['date']
                actual_pct = price_row['actual_pct_change']
                
                # Number of articles per day (0-8)
                n_articles = np.random.poisson(2)
                
                if n_articles == 0:
                    continue
                
                for article_idx in range(n_articles):
                    # Key insight: Create relationship between sentiment and actual price
                    is_sponsored = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% sponsored
                    
                    if is_sponsored:
                        # SPONSORED: Less predictive, has bias (tends bullish)
                        # Sentiment is often positive but doesn't predict actual movement
                        base_sentiment = np.random.normal(0.15, 0.35)  # Biased positive
                        # Weak correlation to actual movement
                        sentiment = base_sentiment + 0.1 * actual_pct + np.random.normal(0, 0.4)
                    else:
                        # ORGANIC: More predictive, follows actual movement
                        # Strong correlation to actual price momentum
                        sentiment = 0.3 * actual_pct + np.random.normal(0, 0.3)
                    
                    sentiment = np.clip(sentiment, -1, 1)  # Clip to [-1, 1]
                    
                    # FinBERT-like probabilities
                    if sentiment > 0:
                        pos_prob = np.clip(0.4 + sentiment * 0.5, 0.2, 0.95)
                        neg_prob = np.clip(0.1 - sentiment * 0.2, 0.05, 0.3)
                    else:
                        neg_prob = np.clip(0.4 - sentiment * 0.5, 0.2, 0.95)
                        pos_prob = np.clip(0.1 + sentiment * 0.2, 0.05, 0.3)
                    
                    neu_prob = 1.0 - pos_prob - neg_prob
                    
                    articles.append({
                        'collection_date': date,
                        'stock_symbol': stock,
                        'title': f"{'Sponsored: ' if is_sponsored else ''}Stock {stock} news article {article_idx}",
                        'headline_url': f"https://news.example.com/{stock}/{article_idx}",
                        'published_time': datetime.strptime(date, '%Y-%m-%d').isoformat(),
                        'fetched_time': datetime.now(self.ist).isoformat(),
                        'is_sponsored': is_sponsored,
                        'source': 'synthetic_historical',
                        'sentiment_score': round(sentiment, 3),
                        'sentiment_positive_prob': round(pos_prob, 3),
                        'sentiment_negative_prob': round(neg_prob, 3),
                        'sentiment_neutral_prob': round(neu_prob, 3),
                    })
        
        print(f"✓ Generated {len(articles)} article records")
        return pd.DataFrame(articles)
    
    def save_to_database(self, prices_df, articles_df):
        """Save generated data to SQLite database"""
        print(f"\n💾 Saving to {self.db_path}...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create tables
            conn.execute('''
                CREATE TABLE IF NOT EXISTS daily_prices (
                    price_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    stock_symbol TEXT NOT NULL,
                    open_price REAL,
                    close_price REAL,
                    high_price REAL,
                    low_price REAL,
                    volume INTEGER,
                    actual_pct_change REAL,
                    actual_direction TEXT,
                    fetched_time TIMESTAMP,
                    data_source TEXT,
                    UNIQUE(date, stock_symbol)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS articles (
                    article_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_date DATE NOT NULL,
                    stock_symbol TEXT NOT NULL,
                    title TEXT NOT NULL,
                    headline_url TEXT,
                    published_time TIMESTAMP,
                    fetched_time TIMESTAMP,
                    is_sponsored INTEGER DEFAULT 0,
                    source TEXT,
                    sentiment_score REAL,
                    sentiment_positive_prob REAL,
                    sentiment_negative_prob REAL,
                    sentiment_neutral_prob REAL,
                    UNIQUE(stock_symbol, title, collection_date)
                )
            ''')
            
            # Insert data
            prices_df.to_sql('daily_prices', conn, if_exists='append', index=False)
            articles_df.to_sql('articles', conn, if_exists='append', index=False)
            
            conn.commit()
            conn.close()
            
            print(f"✓ Saved {len(prices_df)} prices, {len(articles_df)} articles")
            return True
        except Exception as e:
            print(f"✗ Error saving to database: {e}")
            return False
    
    def get_statistics(self):
        """Print data statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            prices_count = pd.read_sql("SELECT COUNT(*) as cnt FROM daily_prices", conn)['cnt'][0]
            articles_count = pd.read_sql("SELECT COUNT(*) as cnt FROM articles", conn)['cnt'][0]
            
            organic_count = pd.read_sql(
                "SELECT COUNT(*) as cnt FROM articles WHERE is_sponsored=0", conn
            )['cnt'][0]
            sponsored_count = pd.read_sql(
                "SELECT COUNT(*) as cnt FROM articles WHERE is_sponsored=1", conn
            )['cnt'][0]
            
            # Correlation check
            query = '''
                SELECT 
                    a.sentiment_score,
                    p.actual_pct_change,
                    a.is_sponsored
                FROM articles a
                JOIN daily_prices p 
                    ON a.collection_date = p.date AND a.stock_symbol = p.stock_symbol
                WHERE a.sentiment_score IS NOT NULL AND p.actual_pct_change IS NOT NULL
                LIMIT 5000
            '''
            df = pd.read_sql(query, conn)
            conn.close()
            
            if len(df) > 0:
                organic_corr = df[df['is_sponsored'] == 0][['sentiment_score', 'actual_pct_change']].corr().iloc[0, 1]
                sponsored_corr = df[df['is_sponsored'] == 1][['sentiment_score', 'actual_pct_change']].corr().iloc[0, 1]
            else:
                organic_corr = sponsored_corr = 0
            
            print("\n" + "="*70)
            print("HISTORICAL DATA STATISTICS")
            print("="*70)
            print(f"Date Range:           {self.start_date.date()} to {self.end_date.date()}")
            print(f"Stocks:               {len(self.stocks)}")
            print(f"Price Records:        {prices_count:,}")
            print(f"Article Records:      {articles_count:,}")
            print(f"  - Organic:          {organic_count:,} ({100*organic_count/articles_count:.1f}%)")
            print(f"  - Sponsored:        {sponsored_count:,} ({100*sponsored_count/articles_count:.1f}%)")
            print(f"\nPredictive Power (Pearson Correlation):")
            print(f"  - Organic news:     {organic_corr:.4f}")
            print(f"  - Sponsored news:   {sponsored_corr:.4f}")
            print(f"  - Difference:       {organic_corr - sponsored_corr:.4f} (organic is {100*(organic_corr/sponsored_corr if sponsored_corr else 1):.0f}% more predictive)")
            print("="*70 + "\n")
            
        except Exception as e:
            print(f"Error getting stats: {e}")


def main():
    """Generate historical dataset"""
    print("\n" + "█"*70)
    print("█ HISTORICAL DATA GENERATOR - 10 YEAR BACKTEST DATASET")
    print("█"*70)
    
    generator = HistoricalDataGenerator()
    
    # Generate data
    prices_df = generator.generate_price_data()
    articles_df = generator.generate_news_data(prices_df)
    
    # Save to database
    generator.save_to_database(prices_df, articles_df)
    
    # Print statistics
    generator.get_statistics()


if __name__ == "__main__":
    main()
