#!/usr/bin/env python3
"""
NIFTY 50 Stock Prediction Engine

Workflow:
1. Collect news sentiment and convert to continuous score [-1, 1]
2. Fetch daily price snapshot
3. Evaluate predicted movement vs actual close movement
4. Generate research and paper-ready tables from history
"""

import os
import sys
from datetime import datetime, timedelta

import pytz
import schedule
import time as time_module

# Add SRC to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SRC"))

from live_data import fetch_daily_snapshot, generate_prediction_report, scan_nifty_50_sentiment
from SRC.past_data import generate_paper_tables, run_research_evaluation

IST = pytz.timezone("Asia/Kolkata")


class StockPredictionEngine:
    """Main orchestrator for stock prediction pipeline."""

    def __init__(self, auto_schedule=False):
        self.auto_schedule = auto_schedule
        self.is_running = False
        self.last_run = None

    def get_time_to_valid_window(self):
        """Get remaining time until next valid news window."""
        now = datetime.now(IST)
        tomorrow_4pm = (now + timedelta(days=1)).replace(hour=16, minute=0, second=0, microsecond=0)

        if now.hour < 9:
            time_left = tomorrow_4pm - now
        elif now.hour >= 16:
            next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            time_left = next_midnight - now
        else:
            today_4pm = now.replace(hour=16, minute=0, second=0, microsecond=0)
            time_left = today_4pm - now
        return time_left

    def print_status(self):
        """Print current operational status."""
        now = datetime.now(IST)
        print("\n" + "=" * 70)
        print("STOCK PREDICTION ENGINE STATUS")
        print("=" * 70)
        print(f"Current Time (IST):    {now.strftime('%A, %d/%m/%Y %H:%M:%S')}")

        if now.hour < 9:
            print("Status:                ACTIVE (Valid News Collection Window)")
        elif now.hour < 16:
            print("Status:                IDLE (Waiting for next news collection)")
        else:
            print("Status:                IDLE (News collection starts at 12am)")

        if self.last_run:
            print(f"Last Completed Run:    {self.last_run}")
        else:
            print("Last Completed Run:    Never")
        print("=" * 70 + "\n")

    def run_sentiment_only(self, sample_size=None, interactive=True):
        """Run only sentiment collection and scoring."""
        print("\n" + "-" * 70)
        print("RUNNING SENTIMENT COLLECTION")
        print("-" * 70)
        sentiment_report = scan_nifty_50_sentiment(
            sample_size=sample_size,
            use_time_filter=True,
            interactive=interactive,
        )
        return sentiment_report is not None

    def run_evaluation_only(self):
        """Run price snapshot and generate prediction evaluation."""
        print("\n" + "-" * 70)
        print("RUNNING END-OF-DAY EVALUATION")
        print("-" * 70)

        price_fetch_success = fetch_daily_snapshot()
        if not price_fetch_success:
            print("Failed to fetch price data. Evaluation aborted.")
            return False

        prediction_success = generate_prediction_report()
        return bool(prediction_success)

    def run_pipeline(self, full_scan=True, sample_size=None, interactive=True):
        """Execute full prediction pipeline."""
        print("\n" + "#" * 70)
        print("# STARTING NIFTY 50 PREDICTION PIPELINE")
        print("#" * 70)

        now = datetime.now(IST)
        print(f"\nStarted at: {now.strftime('%d/%m/%Y %H:%M:%S IST')}\n")

        try:
            if full_scan:
                if not self.run_sentiment_only(sample_size=sample_size, interactive=interactive):
                    print("No sentiment data generated. Skipping predictions.")
                    return False

            if not self.run_evaluation_only():
                return False

            self.last_run = now.strftime("%d/%m/%Y %H:%M:%S IST")
            print("\n" + "#" * 70)
            print("# PIPELINE COMPLETED SUCCESSFULLY")
            print("#" * 70 + "\n")
            return True
        except KeyboardInterrupt:
            print("\nPipeline interrupted by user.")
            return False
        except Exception as error:
            print(f"\nPipeline failed with error: {error}")
            import traceback

            traceback.print_exc()
            return False

    def schedule_automatic_runs(self):
        """Schedule automatic daily runs with separate prediction and evaluation stages."""
        print("\n" + "=" * 70)
        print("SCHEDULING AUTOMATIC RUNS")
        print("=" * 70)
        print("Pipeline will run automatically:")
        print("  - 08:55 IST: collect sentiment and generate daily prediction score")
        print("  - 15:40 IST: fetch close prices and evaluate prediction accuracy")
        print("\nPress Ctrl+C to stop the scheduler.\n")

        schedule.every().day.at("08:55").do(lambda: self.run_sentiment_only(interactive=False))
        schedule.every().day.at("15:40").do(self.run_evaluation_only)

        self.is_running = True
        try:
            while True:
                schedule.run_pending()
                time_module.sleep(60)
        except KeyboardInterrupt:
            print("\nScheduler stopped by user.")
            self.is_running = False


def print_menu():
    """Print interactive menu."""
    print("\n" + "=" * 70)
    print("NIFTY 50 STOCK PREDICTION ENGINE - INTERACTIVE MODE")
    print("=" * 70)
    print("\n1. Run Full Pipeline (Sentiment + Prices + Predictions)")
    print("2. Run Sentiment Analysis Only (scan all 50 companies)")
    print("3. Run Sentiment Analysis - Sample (10 companies for testing)")
    print("4. Run Price Snapshot Only")
    print("5. Run End-of-Day Evaluation (Prices + Predictions)")
    print("6. Schedule Automatic Daily Runs (08:55 + 15:40 IST)")
    print("7. Run Research Evaluation (history + baselines)")
    print("8. Generate Paper Tables")
    print("9. View System Status")
    print("10. Exit\n")


def main():
    """Main entry point."""
    engine = StockPredictionEngine()

    print("\n" + "#" * 70)
    print("# NIFTY 50 STOCK PRICE PREDICTION ENGINE")
    print("# Continuous score output in range [-1, 1]")
    print("#" * 70)

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "run":
            engine.run_pipeline(interactive=False)
        elif arg == "sentiment":
            engine.run_sentiment_only(interactive=False)
        elif arg == "evaluate":
            engine.run_evaluation_only()
        elif arg == "sample":
            engine.run_pipeline(full_scan=True, sample_size=10, interactive=False)
        elif arg == "prices":
            fetch_daily_snapshot()
        elif arg == "schedule":
            engine.schedule_automatic_runs()
        elif arg == "research":
            run_research_evaluation()
        elif arg == "paper":
            generate_paper_tables()
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python app.py [run|sentiment|evaluate|sample|prices|schedule|research|paper]")
    else:
        while True:
            engine.print_status()
            print_menu()
            choice = input("Select option (1-10): ").strip()

            if choice == "1":
                engine.run_pipeline(interactive=True)
            elif choice == "2":
                engine.run_sentiment_only(sample_size=None, interactive=True)
            elif choice == "3":
                engine.run_sentiment_only(sample_size=10, interactive=True)
            elif choice == "4":
                fetch_daily_snapshot()
            elif choice == "5":
                engine.run_evaluation_only()
            elif choice == "6":
                engine.schedule_automatic_runs()
            elif choice == "7":
                run_research_evaluation()
            elif choice == "8":
                generate_paper_tables()
            elif choice == "9":
                pass
            elif choice == "10":
                print("\nExiting... Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
