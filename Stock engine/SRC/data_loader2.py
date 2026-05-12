import yfinance as yf
import pandas as pd
from nsepython import nsefetch
import os

def get_nifty_symbols():
    print("Connecting to NSE for NIFTY 50 list")
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    data = nsefetch(url)
    # Filter out the index name itself
    symbols = [x['symbol'] for x in data['data'] if "NIFTY" not in x['symbol']]
    return symbols

def fetch_daily_snapshot():
    symbols = get_nifty_symbols()
    tickers = [f"{sym}.NS" for sym in symbols]
    
    print(f"Downloading Daily Snapshot for {len(symbols)} companies...")
    
    # Download only the latest 1 day of data
    df = yf.download(tickers, period="1d", interval="1d", group_by='ticker', progress=True)
    
    snapshot_data = []
    
    for sym in symbols:
        try:
            # Extract single stock data
            stock_df = df[f"{sym}.NS"]
            
            if stock_df.empty:
                continue
                
            # Get the latest available row
            today = stock_df.iloc[-1]
            
            # Calculate actual price movement
            open_price = today['Open']
            close_price = today['Close']
            pct_change = ((close_price - open_price) / open_price) * 100
            
            snapshot_data.append({
                "Symbol": sym,
                "Open": round(open_price, 2),
                "Close": round(close_price, 2),
                "Percent_Change": round(pct_change, 2),
                "Actual_Direction": "UP" if pct_change > 0 else "DOWN"
            })
        except:
            continue
            
    # Save to CSV for the next step
    os.makedirs('data', exist_ok=True)
    result_df = pd.DataFrame(snapshot_data)
    result_df.to_csv("daily_prices.csv", index=False)
    print(f"Data saved: daily_prices.csv ({len(result_df)} stocks)")

if __name__ == "__main__":
    fetch_daily_snapshot()