# CONTINUOUS STOCK PREDICTION ENGINE - ARCHITECTURE v3.0

## 📊 EXECUTIVE SUMMARY

Transform binary UP/DOWN predictions into continuous prediction scores in the range **[-1, 1]**:
- **+1** = expected heavy gain (bullish)
- **0** = neutral/no meaningful move
- **-1** = expected heavy fall (bearish)

Key enhancements:
- Article-level sentiment → stock-level aggregated score
- Robust normalization (tanh-based)
- Sponsored vs non-sponsored tracking
- Continuous evaluation metrics (correlation, MAE, RMSE)
- SQLite persistence with idempotent scheduling
- Backward-compatible CLI

---

## 🏗️ UPDATED SYSTEM ARCHITECTURE

```
NEWS COLLECTION (4pm-9am IST)
    ↓
    [GoogleNews] → Articles with timestamps
    ↓
ARTICLE-LEVEL PROCESSING
    ├─ [Sponsored Detection] → Identify ads/sponsored content
    └─ [FinBERT Sentiment] → Article score ∈ [-1, 1]
    ↓
STOCK-LEVEL AGGREGATION
    ├─ Raw signal = mean(all article scores)
    ├─ Compute separate aggregates:
    │  ├─ all_sentiment (all articles)
    │  ├─ sponsored_sentiment (sponsored only)
    │  └─ organic_sentiment (non-sponsored only)
    └─ Store article metadata in DB
    ↓
CONTINUOUS SCORE GENERATION
    ├─ Apply robust normalization: pred_score = tanh(raw_signal * k)
    └─ Map to expected % change: expected_pct = pred_score * calibration_k
    ↓
OUTPUT & EVALUATION
    ├─ pred_score ∈ [-1, 1]
    ├─ pred_direction (BULLISH/NEUTRAL/BEARISH) from threshold
    ├─ expected_pct_change (percent)
    ├─ Compare vs actual close-to-close return
    └─ Evaluate on continuous metrics: Pearson corr, RMSE, MAE
    ↓
PERSISTENCE & REPORTING
    └─ SQLite storage + daily/weekly/monthly reports
```

---

## 📋 DATA SCHEMA (SQLite)

### Table 1: `articles`
```sql
CREATE TABLE articles (
    article_id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_date DATE,
    stock_symbol TEXT NOT NULL,
    title TEXT NOT NULL,
    headline_url TEXT,
    article_text TEXT,
    published_time TIMESTAMP,
    fetched_time TIMESTAMP,
    
    -- Metadata
    is_sponsored INTEGER DEFAULT 0,  -- 1 if sponsored/ad
    source TEXT,                      -- 'GoogleNews', etc.
    
    -- Sentiment (FinBERT output)
    sentiment_score REAL,              -- Raw output from FinBERT [-1, 1]
    sentiment_positive_prob REAL,
    sentiment_negative_prob REAL,
    sentiment_neutral_prob REAL,
    
    UNIQUE(stock_symbol, article_id, collection_date)
);
CREATE INDEX idx_articles_date_symbol ON articles(collection_date, stock_symbol);
CREATE INDEX idx_articles_sponsored ON articles(is_sponsored);
```

### Table 2: `daily_signals`
```sql
CREATE TABLE daily_signals (
    signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_date DATE,
    stock_symbol TEXT NOT NULL,
    
    -- Raw aggregates
    article_count INTEGER,
    sponsored_count INTEGER,
    organic_count INTEGER,
    
    -- Aggregated sentiment (raw mean before normalization)
    all_sentiment_raw REAL,        -- mean of all article scores
    sponsored_sentiment_raw REAL,  -- mean of sponsored only
    organic_sentiment_raw REAL,    -- mean of non-sponsored only
    
    -- Normalized scores [-1, 1]
    pred_score REAL,               -- final prediction score [-1, 1]
    sponsored_pred_score REAL,     -- prediction from sponsored only
    organic_pred_score REAL,       -- prediction from organic only
    
    -- Direction (derived from score)
    pred_direction TEXT,           -- 'BULLISH', 'NEUTRAL', 'BEARISH'
    direction_threshold REAL,      -- threshold used (default ±0.1)
    
    -- Expected move
    expected_pct_change REAL,      -- k * pred_score
    calibration_k REAL,            -- scaling factor (learned from history)
    
    -- Metadata
    processing_time TIMESTAMP,
    is_production INTEGER DEFAULT 1,  -- 0 if test/sample run
    
    UNIQUE(collection_date, stock_symbol)
);
CREATE INDEX idx_signals_date_symbol ON daily_signals(collection_date, stock_symbol);
```

### Table 3: `daily_prices`
```sql
CREATE TABLE daily_prices (
    price_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    stock_symbol TEXT NOT NULL,
    
    open_price REAL,
    close_price REAL,
    high_price REAL,
    low_price REAL,
    volume INTEGER,
    
    -- Actual movement (target variable)
    actual_pct_change REAL,    -- (close - open) / open * 100
    actual_direction TEXT,     -- 'UP', 'DOWN', 'NEUTRAL' (if |pct| < 0.5%)
    
    -- Metadata
    fetched_time TIMESTAMP,
    data_source TEXT,          -- 'yfinance', 'NSE_API'
    
    UNIQUE(date, stock_symbol)
);
CREATE INDEX idx_prices_date_symbol ON daily_prices(date, stock_symbol);
```

### Table 4: `evaluations`
```sql
CREATE TABLE evaluations (
    eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
    eval_date DATE,
    lookback_days INTEGER,  -- e.g., 7 (weekly), 30 (monthly)
    
    -- Metrics
    pearson_corr REAL,              -- Corr(pred_score, actual_pct_change)
    spearman_corr REAL,             -- Rank correlation
    mae REAL,                        -- Mean Absolute Error (pred_pct vs actual_pct)
    rmse REAL,                       -- Root Mean Squared Error
    mape REAL,                       -- Mean Absolute Percentage Error
    directional_accuracy REAL,       -- % of correct sign predictions
    
    -- Subset metrics
    sponsored_pearson_corr REAL,
    organic_pearson_corr REAL,
    
    -- Data
    sample_count INTEGER,           -- Predictions evaluated
    
    UNIQUE(eval_date, lookback_days)
);
```

### Table 5: `calibration_history`
```sql
CREATE TABLE calibration_history (
    cal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    calibration_date DATE,
    lookback_days INTEGER,          -- Train window (e.g., last 90 days)
    
    -- Learned parameters
    k_optimal REAL,                 -- Best scaling for % conversion
    threshold_bullish REAL,         -- Score threshold for BULLISH (default +0.1)
    threshold_bearish REAL,         -- Score threshold for BEARISH (default -0.1)
    normalization_param REAL,       -- Alpha in tanh(alpha * x)
    
    -- Fit quality
    rmse_on_train REAL,
    rmse_on_test REAL,              -- Hold-out set
    
    is_active INTEGER DEFAULT 1,    -- 1 if currently used
    created_time TIMESTAMP
);
```

---

## 🧮 SCORING FORMULA

### Step 1: Article-level Sentiment (FinBERT)
```
s_article = E[P_positive] - E[P_negative]
where P_* are FinBERT softmax probabilities
Range: (-1, 1) approximately
```

### Step 2: Stock-level Raw Signal (Aggregation)
```
raw_signal = mean(s_article_1, s_article_2, ..., s_article_n)

For sponsored tracking:
- raw_signal_organic = mean(s_i for non-sponsored articles)
- raw_signal_sponsored = mean(s_i for sponsored articles)
```

### Step 3: Robust Normalization → [-1, 1]
```
Option A: Tanh-based (RECOMMENDED for non-Gaussian)
    pred_score = tanh(α * raw_signal)
    where α ≈ 1.5 − 2.0 (learned/calibrated)
    
Option B: Clipped Z-score (if Gaussian)
    z = (raw_signal - μ) / σ
    pred_score = clip(z / 3, -1, +1)
    
We use Option A by default.
```

### Step 4: Direction Classification
```
pred_direction = 
    BULLISH   if pred_score > +τ      (τ default = +0.1)
    BEARISH   if pred_score < -τ      (τ default = -0.1)
    NEUTRAL   otherwise
```

### Step 5: Expected % Move (Optional)
```
expected_pct_change = k * pred_score

Where k is learned from historical data:
    k = argmin Σ(expected_pct_i - actual_pct_i)²
    over training window (e.g., last 90 days)
    
Typical range for k: 3-8 (% move per unit score)
```

---

## 🗂️ IMPLEMENTATION ROADMAP

### STEP 1: Database Layer & Scoring Engine [NEW]
- [x] Create SQLite schema (`db_schema.py`)
- [x] Build `ScoringEngine` class with normalization functions
- [x] Implement article→stock aggregation logic
- [x] Build calibration module (learn k from history)
- [ ] **→ Output**: `SRC/scoring_engine.py`, `SRC/db_schema.py`, sample calibration plots

### STEP 2: Refactor Data Pipeline [2-3 files modified]
- [ ] Enhance `Sentiment.py` to:
  - Detect sponsored articles
  - Return full FinBERT probabilities
  - Store articles in DB
- [ ] Modify `Scanner.py` to:
  - Call enhanced Sentiment
  - Save article-level records
  - Compute stock-level aggregates
- [ ] Enhance `data_loader2.py` to:
  - Fetch price data into DB
  - Compute actual_direction (3-class: UP/DOWN/NEUTRAL)
- [ ] **→ Output**: Updated `SRC/Sentiment.py`, `SRC/Scanner.py`, `SRC/data_loader2.py`

### STEP 3: New Scoring Pipeline [NEW]
- [ ] Create `SRC/scoring_pipeline.py`:
  - Load articles from DB
  - Aggregate per stock
  - Apply normalization
  - Compute pred_score & pred_direction
  - Store in `daily_signals` table
- [ ] **→ Output**: `SRC/scoring_pipeline.py`

### STEP 4: Evaluation & Reporting [NEW]
- [ ] Create `SRC/evaluation_metrics.py`:
  - Pearson/Spearman correlation
  - MAE, RMSE, MAPE
  - Directional accuracy
  - Per-subset metrics (sponsored vs organic)
- [ ] Create `SRC/calibration_learner.py`:
  - Train k and thresholds
  - Cross-validate on hold-out sets
  - Store in `calibration_history` table
- [ ] Create `SRC/reporting.py`:
  - Daily tables (HTML/CSV)
  - Weekly rolling window reports
  - Monthly aggregates + charts
  - Sponsored vs organic comparison plots
- [ ] **→ Output**: 3 new modules + HTML/CSV reports

### STEP 5: Update Orchestrator [app.py]
- [ ] Add new CLI commands:
  - `python app.py calibrate` → learn k and thresholds
  - `python app.py score` → generate pred_scores
  - `python app.py evaluate --days=7` → compute metrics
  - `python app.py report --period=daily` → generate reports
- [ ] Maintain backward compatibility:
  - Old `run`, `sample`, `prices` still work
  - Transparently insert into DB
- [ ] Add scheduled runs via `schedule` + DB idempotency
- [ ] **→ Output**: Enhanced `app.py`

### STEP 6: Calibration & Backtesting [NEW]
- [ ] Build `backtester.py`:
  - Replay historical signals vs prices
  - Plot cumulative P&L if k applied
  - Suggest optimal k values
- [ ] Generate calibration report with correlation curves

---

## 🎯 OUTPUT EXAMPLES

### Daily Prediction CSV
```
date,symbol,article_count,all_sentiment_raw,pred_score,pred_direction,expected_pct_change,actual_pct_change,result
2024-04-15,TCS,5,0.32,0.31,BULLISH,2.48,2.15,✓ CORRECT
2024-04-15,INFY,3,-0.18,-0.18,BEARISH,-1.44,-1.89,✓ CORRECT
2024-04-15,RELIANCE,2,0.05,0.05,NEUTRAL,0.40,0.82,✓ CLOSE
```

### Evaluation Report (Weekly)
```
Lookback: 7 days
Samples: 245 predictions

Pearson Correlation:        0.42 (moderate)
Spearman Correlation:       0.38
MAE (actual % vs expected): 2.3%
RMSE:                       3.1%
MAPE:                       8.2%
Directional Accuracy:       58.2%

Sponsored only:
  - Pearson Corr: 0.35
  - Sample count: 95

Non-Sponsored:
  - Pearson Corr: 0.46
  - Sample count: 150
  - [Conclusion: Organic news more predictive]
```

---

## 🔄 BACKWARD COMPATIBILITY

Current CLI commands:
```bash
python app.py run              # Full pipeline
python app.py sample           # Sample 10
python app.py prices           # Price fetch only
python app.py schedule         # Scheduler
```

**All remain unchanged.** Internally:
- Articles saved to DB (new)
- Predictions computed as continuous scores (new)
- Direction derived from threshold (backward compatible view)
- Historical data available for calibration (new)

---

## 💾 DEPENDENCIES (NEW)

```
sqlalchemy>=2.0.0          # ORM for DB operations
scikit-learn>=1.3.0        # Correlation, calibration functions
scipy>=1.10.0              # Already present
matplotlib>=3.7.0          # New (for calibration plots)
seaborn>=0.12.0            # New (for reporting charts)
```

---

## 📅 NEXT STEPS

1. **NOW**: Print this architecture → stakeholder review
2. **STEP 1**: Implement database schema + scoring engine
3. **STEP 2**: Refactor sentim pipeline
4. **STEP 3-6**: Progressive enhancement

---

## 🚀 SUCCESS CRITERIA

✅ Continuous pred_score ∈ [-1, 1] is primary output
✅ Correlation(pred_score, actual_pct_change) > 0.35
✅ Database persistence + idempotent reruns
✅ Sponsored vs organic comparison available
✅ CLI backward compatible
✅ Daily + weekly + monthly reports automated
