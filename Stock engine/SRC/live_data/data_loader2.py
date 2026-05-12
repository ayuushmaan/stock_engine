from datetime import datetime
import os

import pandas as pd
import pytz
import yfinance as yf
from nsepython import nsefetch

from SRC.common.paths import HISTORY_DIR, LATEST_DIR, ensure_dirs

def get_nifty_symbols_with_sector():
    """Fetch Nifty 50 symbols and sector metadata from NSE."""
    print("Connecting to NSE for NIFTY 50 list...")
    try:
        url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
        data = nsefetch(url)
        rows = []
        for item in data["data"]:
            symbol = item.get("symbol")
            if not symbol or "NIFTY" in symbol:
                continue
            meta = item.get("meta", {})
            sector = meta.get("industry", "UNKNOWN") if isinstance(meta, dict) else item.get("industry", "UNKNOWN")
            rows.append({"Symbol": symbol, "Sector": sector or "UNKNOWN"})
        frame = pd.DataFrame(rows).drop_duplicates(subset=["Symbol"])
        return frame.sort_values("Symbol").reset_index(drop=True)
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return pd.DataFrame(columns=["Symbol", "Sector"])


def fetch_daily_snapshot():
    """
    Fetch daily price snapshot for all Nifty 50 companies.
    Includes date + sector and appends to historical price file for research validation.
    """
    symbol_frame = get_nifty_symbols_with_sector()
    if symbol_frame.empty:
        print("Failed to fetch symbols. Aborting.")
        return False

    symbols = symbol_frame["Symbol"].tolist()
    tickers = [f"{sym}.NS" for sym in symbols]
    run_date = datetime.now(pytz.timezone("Asia/Kolkata")).date().isoformat()

    print(f"Downloading Daily Snapshot for {len(symbols)} Nifty 50 companies...")
    print(f"Timestamp: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d/%m/%Y %H:%M:%S %Z')}")

    try:
        df = yf.download(tickers, period="1d", interval="1d", group_by="ticker", progress=False)
        snapshot_data = []
        successful = 0

        for row in symbol_frame.itertuples(index=False):
            sym = row.Symbol
            sector = row.Sector
            try:
                stock_df = df[f"{sym}.NS"]
                if stock_df.empty:
                    continue
                today = stock_df.iloc[-1]
                open_price = float(today["Open"])
                close_price = float(today["Close"])
                high_price = float(today["High"])
                low_price = float(today["Low"])
                volume = int(today["Volume"])
                pct_change = ((close_price - open_price) / open_price) * 100 if open_price else 0.0
                snapshot_data.append(
                    {
                        "Date": run_date,
                        "Symbol": sym,
                        "Sector": sector,
                        "Open": round(open_price, 2),
                        "Close": round(close_price, 2),
                        "High": round(high_price, 2),
                        "Low": round(low_price, 2),
                        "Volume": volume,
                        "Percent_Change": round(pct_change, 6),
                        "Actual_Direction": "UP" if pct_change > 0 else "DOWN",
                    }
                )
                successful += 1
            except Exception as e:
                print(f"  Warning: Could not fetch data for {sym}: {str(e)[:40]}")
                continue

        result_df = pd.DataFrame(snapshot_data)
        ensure_dirs()
        latest_path = os.path.join(LATEST_DIR, "daily_prices.csv")
        result_df.to_csv(latest_path, index=False)

        history_path = os.path.join(HISTORY_DIR, "daily_prices_history.csv")
        if pd.io.common.file_exists(history_path):
            existing = pd.read_csv(history_path)
            merged = pd.concat([existing, result_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["Date", "Symbol"], keep="last")
        else:
            merged = result_df.copy()
        merged.to_csv(history_path, index=False)

        print(f"\nDaily prices saved: {latest_path} ({successful}/{len(symbols)} stocks)")
        print(f"History updated in '{history_path}'")
        return True
    except Exception as e:
        print(f"Error downloading data: {e}")
        return False


if __name__ == "__main__":
    fetch_daily_snapshot()
