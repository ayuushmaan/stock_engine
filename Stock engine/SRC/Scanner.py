from nsepython import *
from Sentiment import SentimentEngine
import pandas as pd

def get_nifty_100_list():
    print("Connecting to NSE for NIFTY 50 list")
    try:
        url = 'https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050'
        positions = nsefetch(url)
        df = pd.DataFrame(positions['data'])
        
        # CRITICAL FIX: Filter out the "NIFTY 100" index entry itself
        # We only want rows where the symbol is NOT containing 'NIFTY'
        nifty_stocks = df[~df['symbol'].str.contains("NIFTY")]
        
        return nifty_stocks
    except Exception as e:
        print(f"Error fetching Nifty list: {e}")
        return pd.DataFrame()

# 1. Get the Cleaned List
df = get_nifty_100_list()

if df.empty:
    print("Failed to get stock list. Check internet or NSE connection.")
    exit()

# Randomly sample 10 stocks for a quick test (remove .sample() to run ALL 100)
# running 100 stocks takes about 5-8 minutes
Sampled_df = df.sample(n=10, random_state=1) 

# 2. Initialize Engine
engine = SentimentEngine()
signals = []

print(f"Scanning {len(Sampled_df)} NIFTY 100 companies for news...")

# 3. Validated Loop
for sym in Sampled_df['symbol']:
    print(f"Searching: {sym}...", end=" ")
    
    news = engine.get_headlines(sym)
    
    if news:
        score = engine.analyze_sentiment(news)
        print(f"Score: {score:.3f}")
        
        # Save signal if it is Significant (Positive OR Negative)
        if abs(score) > 0.05: 
            signals.append({
                "Symbol": sym, 
                "Sentiment_Score": round(score, 3),
                "Direction": "BULLISH" if score > 0 else "BEARISH"
            })
    else:
        print("No News")

# 4. Final Report
if signals:
    report = pd.DataFrame(signals)
    print("\n" + "="*40)
    print("FINAL NIFTY 100 SENTIMENT REPORT")
    print("="*40)
    print(report.sort_values(by="Sentiment_Score", ascending=False))
    
    # Save to CSV for the next step (Prediction)
    report.to_csv("nifty_sentiment_results.csv", index=False)
    print(" Saved results to 'nifty_sentiment_results.csv'")
else:
    print("No significant sentiment found in this batch.")