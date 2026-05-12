# NIFTY 50 Stock Price Prediction Engine v2.0

## Clean Folder Layout (Current)

```
Stock engine/
|-- app.py
|-- train_ml_model.py        # historical training pipeline
|-- quick_demo_ml.py         # small training demo
|-- requirements.txt
|-- SRC/
|   |-- live_data/          # daily fetch + live prediction flow
|   |-- past_data/          # historical training + backtesting flow
|   `-- common/             # shared utilities and model config loading
|-- scripts/                # Task Scheduler runners + init
|-- outputs/
|   |-- latest/             # latest run outputs
|   |-- history/            # accumulated daily history
|   |-- research/           # research evaluation outputs
|   `-- paper/              # paper-ready tables
|-- db/                     # SQLite DB files
|-- docs/                   # notes + architecture docs
`-- market_env/             # virtual environment
```

See `docs/CODEBASE_LAYOUT.md` for the past-data vs live-data split.

## Updated Output Paths

- `outputs/latest/daily_prices.csv`
- `outputs/latest/nifty_sentiment_results.csv`
- `outputs/latest/prediction_report.csv`
- History:
  - `outputs/history/daily_prices_history.csv`
  - `outputs/history/nifty_sentiment_history.csv`
  - `outputs/history/prediction_report_history.csv`

## Useful Commands

- `python app.py run`
- `python app.py sentiment`
- `python app.py evaluate`
- `python app.py research`
- `python app.py paper`

## 🎯 Optimization Overview

This optimized version of your stock prediction engine is designed to:

1. **Fetch news only during 4pm-9am window** - Market close to market open period when maximum trading-relevant news is available
2. **Analyze sentiment** using FinBERT (financial BERT model) on collected news
3. **Predict stock price movements** for all Nifty 50 companies based on sentiment analysis
4. **Measure accuracy** by comparing predictions against actual price movements

---

## ⏰ Valid News Collection Window: 4pm - 9am (IST)

### Why This Window?
- **4:00 PM**: Market closes (after-trading-hours news starts)
- **9:00 AM**: Market opens next day (before-market-open analysis complete)
- This window captures all overnight news that could impact trading

### System Behavior:
- ✅ **12:00 AM - 9:00 AM**: News collection **ACTIVE** (valid window open)
- ⏸ **9:00 AM - 4:00 PM**: System **IDLE** (no valid news window)
- ⏸ **4:00 PM onwards**: System **IDLE** until next midnight

---

## 📋 Project Structure

```
Stock engine/
├── app.py                          # Main orchestrator (NEW)
├── requirements.txt                # Python dependencies
├── daily_prices.csv               # Output: Daily price data
├── nifty_sentiment_results.csv    # Output: Sentiment analysis results
├── prediction_report.csv          # Output: Final predictions
│
├── SRC/
│   ├── data_loader2.py            # Enhanced: Fetches Nifty 50 price data
│   ├── Scanner.py                 # Enhanced: Scans news + time filtering
│   ├── Sentiment.py               # Enhanced: FinBERT analysis + time window
│   └── Predictions.py             # Enhanced: Generates accuracy reports
│
└── market_env/                    # Virtual environment (your existing setup)
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
# Activate your virtual environment
market_env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Run the Application

**Option A: Interactive Mode** (Recommended for first-time)
```bash
python app.py
```
This opens an interactive menu where you can:
1. Run Full Pipeline (News → Sentiment → Prices → Predictions)
2. Run Sentiment Analysis Only
3. Run Sentiment Analysis with Sample (10 companies for testing)
4. Fetch Daily Prices Only
5. Generate Predictions from Existing Data
6. Schedule Automatic Runs
7. View System Status

**Option B: Command Line (Quick execution)**
```bash
# Full pipeline - all 50 companies
python app.py run

# Quick test - 10 companies only
python app.py sample

# Fetch prices only
python app.py prices

# Automatic scheduling (runs at midnight & 9am)
python app.py schedule
```

---

## 📊 Workflow: How It Works

### Automated Pipeline Flow:
```
┌─────────────────────────┐
│  Check Time Window      │
│  (4pm-9am valid?)       │
└────────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ Fetch News for      │
    │ Nifty 50 Companies  │
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ Analyze Sentiment   │
    │ (FinBERT Model)     │
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ Fetch Daily Prices  │
    │ (Yesterday's close) │
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ Generate Predictions│
    │ Compare vs Actual   │
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ Generate Report     │
    │ Calculate Accuracy  │
    └─────────────────────┘
```

---

## 📁 Key Enhancements & Changes

### ✨ New Features:

1. **Time-Based Filtering** (Sentiment.py)
   - Automatically filters news to 4pm-9am window
   - Checks IST timezone to ensure correct window
   - Provides time status warnings

2. **Enhanced Scanner** (Scanner.py)
   - Shows time window validity
   - Includes all 50 stocks (not just samples)
   - Better progress tracking with indices
   - Improved error handling

3. **Optimized Data Loader** (data_loader2.py)
   - Includes high/low prices
   - Tracks volume data
   - Better error reporting

4. **Main Orchestrator** (app.py) - **NEW**
   - Single entry point for all operations
   - Interactive menu system
   - Automatic scheduling support
   - Real-time status monitoring

5. **Enhanced Predictions** (Predictions.py)
   - Detailed accuracy metrics
   - Signal breakdown (Bullish/Bearish)
   - Comprehensive reports
   - CSV output for analysis

### 🔧 Optimization Details:

| Component | Optimization |
|-----------|-------------|
| Time Management | Added IST timezone checks + 4pm-9am window validation |
| Error Handling | Comprehensive try-catch blocks with user feedback |
| Progress Tracking | Shows current status, counts, timestamps |
| Output Quality | Enhanced CSV reports with additional metrics |
| Performance | Parallel processing ready, progress bars |

---

## 📊 Output Files Explained

### 1. `daily_prices.csv`
```
Symbol,Open,Close,High,Low,Volume,Percent_Change,Actual_Direction
RELIANCE,2500.00,2510.00,2515.00,2490.00,1000000,0.4,UP
INFY,1800.00,1795.00,1805.00,1790.00,500000,-0.28,DOWN
```

### 2. `nifty_sentiment_results.csv`
```
Symbol,Sentiment_Score,Direction,Headline_Count,Timestamp
INFY,0.125,BULLISH,5,2024-03-12T08:30:00+05:30
TCS,-0.085,BEARISH,3,2024-03-12T08:31:00+05:30
```

### 3. `prediction_report.csv`
```
Symbol,Prediction,Score,Actual,Correct,Change%
INFY,UP,0.125,UP,✓,2.15
TCS,DOWN,-0.085,UP,✗,-1.50
```

---

## ⏰ Scheduling Setup (Optional)

For automated runs every day:

```bash
# Start scheduler (runs at midnight & 9am IST)
python app.py schedule
```

Or set up Windows Task Scheduler:
1. Open Task Scheduler
2. Create Basic Task
3. Trigger: Daily at 00:00 (Midnight IST)
4. Action: Run `python c:\path\to\app.py run`

---

## 🐛 Troubleshooting

### Issue: "No News Found" for all stocks
**Solution**: Check internet connection, ensure you're within 4pm-9am window

### Issue: "NSE Connection Error"
**Solution**: NSE website might be down, try again later

### Issue: "Module not found" errors
**Solution**: 
```bash
pip install -r requirements.txt
```

### Issue: "Outside valid news window" warning
**Solution**: This is normal. Either:
- Wait until midnight (4pm-9am window opens)
- Override by selecting option in interactive menu
- Set `use_time_filter=False` for testing

---

## 📈 Next Steps for Further Optimization

1. **Database Integration**: Store predictions in SQLite for historical analysis
2. **Real-time Alerting**: Push notifications when sentiment is very strong
3. **Multi-day Analysis**: Track prediction accuracy over weeks/months
4. **Additional Signals**: Combine sentiment with technical indicators (RSI, MACD)
5. **Paper Trading**: Simulate trades based on predictions

---

## 📝 Key Modifications Summary

| File | Changes |
|------|---------|
| **Sentiment.py** | +Time filtering, IST timezone support, window validation |
| **Scanner.py** | +Time window checks, better output, full 50 companies |
| **data_loader2.py** | +Timestamp tracking, better error handling |
| **Predictions.py** | +Detailed metrics, improved reporting, CSV export |
| **app.py** | 🆕 NEW: Main orchestrator with scheduling |
| **requirements.txt** | +Added pytz, schedule, and finalized dependencies |

---

## 💡 Usage Tips

1. **First Run**: Use `python app.py sample` to test with 10 companies
2. **Full Analysis**: Run `python app.py run` to scan all 50 companies
3. **Best Time**: Run between 4pm-9am IST for complete news window
4. **Incremental**: Save outputs for later analysis and comparison
5. **Scheduling**: Use OS scheduler for hands-off automated runs

---

## 🎓 Educational Note

The 4pm-9am window is optimal because:
- Markets close at 3:30 PM (trading ends)
- News breaks 4pm-9am (overnight period)
- New traders analyze this news before 9am market open
- Price predictions made at 9am can be verified by end of day

---

**Version**: 2.0 (Optimized for 4pm-9am window)  
**Last Updated**: March 2026  
**Status**: Production Ready

## Research Mode (Continuous Score)

This project now supports continuous prediction scores in `[-1, 1]`:
- `+1` means expected heavy gain
- `0` means neutral
- `-1` means expected heavy fall

### Exact scoring formula

1. Article-level score:
   - `s_article = P(positive) - P(negative)` from FinBERT softmax
2. Stock raw signal:
   - `raw_signal = mean(s_article_1 ... s_article_n)`
3. Bounded prediction score:
   - `pred_score = tanh(alpha * raw_signal)`
   - default `alpha = 1.8`
4. Direction thresholds:
   - `pred_score > +0.10` => BULLISH
   - `pred_score < -0.10` => BEARISH
   - otherwise => NEUTRAL
5. Price move estimate:
   - `expected_pct_change = k * pred_score`
   - `k` calibrated by least squares from historical data (train split only)

### Why these defaults

- `tanh` keeps predictions bounded and robust to outlier news bursts.
- `alpha=1.8` increases separation without saturating too early.
- `0.10` threshold reduces noisy near-zero directional calls.
- Calibrated `k` converts score units to expected return units without leakage.

### New history files

- `nifty_sentiment_history.csv`
- `daily_prices_history.csv`
- `prediction_report_history.csv`

### New research command

Run strict time-split evaluation and baselines:

```bash
python app.py research
```

Outputs:
- `research_summary.csv` (main model + baselines)
- `research_robustness.csv` (sector, regime, threshold checks)
- `research_sponsored_vs_non_sponsored.csv`
- `research_repro_manifest.json` (seed, hashes, split metadata)

### Baselines included

- Random baseline
- Market return constant baseline
- Simple momentum baseline (last symbol return)
- Headline count baseline (train-fit linear model)
