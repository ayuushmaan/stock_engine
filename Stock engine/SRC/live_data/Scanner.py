from datetime import datetime
import os

import pandas as pd
import pytz
from nsepython import nsefetch

from SRC.common.paths import HISTORY_DIR, LATEST_DIR, ensure_dirs
from live_data.Sentiment import SentimentEngine


def get_nifty_50_list():
    """Fetch current Nifty 50 stock symbols and sector metadata."""
    print("Connecting to NSE for NIFTY 50 list")
    try:
        url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
        positions = nsefetch(url)
        df = pd.DataFrame(positions["data"])
        nifty_stocks = df[~df["symbol"].str.contains("NIFTY", na=False)].copy()
        if "meta" in nifty_stocks.columns:
            nifty_stocks["Sector"] = nifty_stocks["meta"].apply(
                lambda value: value.get("industry", "UNKNOWN") if isinstance(value, dict) else "UNKNOWN"
            )
        elif "industry" in nifty_stocks.columns:
            nifty_stocks["Sector"] = nifty_stocks["industry"].fillna("UNKNOWN")
        else:
            nifty_stocks["Sector"] = "UNKNOWN"
        return nifty_stocks
    except Exception as e:
        print(f"Error fetching Nifty list: {e}")
        return pd.DataFrame()


def check_valid_news_window():
    """Check if current time is within 4pm-9am valid window."""
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    if now.hour < 9:
        return True, f"Within valid news window (current time: {now.strftime('%H:%M %Z')})"
    return False, f"Outside valid news window (current time: {now.strftime('%H:%M %Z')}). Next window: Today 4pm to Tomorrow 9am"


def scan_nifty_50_sentiment(sample_size=None, use_time_filter=True, interactive=True):
    """
    Scan Nifty 50 companies and output continuous sentiment predictions.

    Pred_Score is in [-1, 1]:
    +1 => heavy expected gain, -1 => heavy expected fall.
    """
    is_valid, status_msg = check_valid_news_window()
    print(f"\n{status_msg}")
    print("=" * 60)

    if not is_valid and use_time_filter:
        print("Warning: running outside optimal news collection window.")
        print("News fetched may not cover the complete 4pm-9am period.")
        if interactive:
            response = input("Continue anyway? (y/n): ")
            if response.lower() != "y":
                print("Scan cancelled.")
                return None
        else:
            print("Non-interactive mode enabled. Continuing automatically.")

    df = get_nifty_50_list()
    if df.empty:
        print("Failed to get stock list. Check internet or NSE connection.")
        return None

    print(f"\nTotal Nifty 50 stocks available: {len(df)}")
    if sample_size and sample_size < len(df):
        sampled_df = df.sample(n=sample_size, random_state=1)
        print(f"Sampling {sample_size} stocks for testing...")
    else:
        sampled_df = df
        print(f"Scanning ALL {len(df)} Nifty 50 companies...")

    engine = SentimentEngine(use_time_filter=use_time_filter)
    signals = []

    window_start, window_end = engine.get_valid_news_window()
    print(f"Fetching news from {window_start.strftime('%d/%m %H:%M')} to {window_end.strftime('%d/%m %H:%M')}")
    print("=" * 60)

    run_date = datetime.now(pytz.timezone("Asia/Kolkata")).date().isoformat()
    for idx, row in enumerate(sampled_df.itertuples(index=False), 1):
        sym = getattr(row, "symbol")
        sector = getattr(row, "Sector", "UNKNOWN")
        print(f"[{idx}/{len(sampled_df)}] Searching: {sym}...", end=" ", flush=True)
        try:
            articles = engine.get_news_results(sym)
            if not articles:
                print("No News Found")
                continue

            analysis = engine.analyze_articles(articles)
            pred_score = analysis["pred_score"]
            print(
                f"Score: {pred_score:>7.3f} ({len(articles)} articles, "
                f"SP:{analysis['sponsored_count']}, NSP:{analysis['non_sponsored_count']})"
            )

            signals.append(
                {
                    "Date": run_date,
                    "Symbol": sym,
                    "Sector": sector,
                    "Sentiment_Score_Raw": round(analysis["all_sentiment_raw"], 6),
                    "Pred_Score": round(pred_score, 6),
                    "Direction": analysis["pred_direction"],
                    "Intensity": analysis["pred_intensity"],
                    "Headline_Count": len(articles),
                    "Sponsored_Count": analysis["sponsored_count"],
                    "NonSponsored_Count": analysis["non_sponsored_count"],
                    "Sponsored_Sentiment_Raw": analysis["sponsored_sentiment_raw"],
                    "NonSponsored_Sentiment_Raw": analysis["non_sponsored_sentiment_raw"],
                    "Sponsored_Pred_Score": analysis["sponsored_pred_score"],
                    "NonSponsored_Pred_Score": analysis["non_sponsored_pred_score"],
                    "Sponsored_Penalty_Applied": analysis["sponsored_penalty"],
                    "Model_Source": analysis["model_source"],
                    "Timestamp": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat(),
                }
            )
        except Exception as e:
            print(f"Error: {str(e)[:60]}")
            continue

    print("\n" + "=" * 60)
    if not signals:
        print("No sentiment signals found in this batch.")
        return None

    report = pd.DataFrame(signals)
    report["Sentiment_Score"] = report["Pred_Score"]

    ensure_dirs()
    latest_path = os.path.join(LATEST_DIR, "nifty_sentiment_results.csv")
    report.to_csv(latest_path, index=False)

    history_path = os.path.join(HISTORY_DIR, "nifty_sentiment_history.csv")
    if pd.io.common.file_exists(history_path):
        existing = pd.read_csv(history_path)
        merged = pd.concat([existing, report], ignore_index=True)
        merged = merged.drop_duplicates(subset=["Date", "Symbol"], keep="last")
    else:
        merged = report.copy()
    merged.to_csv(history_path, index=False)

    print("FINAL NIFTY 50 CONTINUOUS SENTIMENT REPORT")
    print(f"Signals Found: {len(report)}")
    print("=" * 60)
    print(
        report.sort_values(by="Pred_Score", ascending=False)[
            ["Symbol", "Pred_Score", "Direction", "Headline_Count", "Sponsored_Count"]
        ].to_string(index=False)
    )
    print(f"\nResults saved to '{latest_path}'")
    print(f"History updated in '{history_path}'")
    return report


if __name__ == "__main__":
    scan_nifty_50_sentiment(sample_size=None, use_time_filter=True, interactive=True)
