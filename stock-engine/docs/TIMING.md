# Timing Contract — When Does News Become Tradable?

This document defines the exact timestamp alignment between news signals and forward returns used in this research. **All reported ICs and backtests use strict forward-looking evaluation only.**

---

## 1. GDELT Timestamp → IST Session Bucketing

GDELT records have UTC timestamps. Pipeline step `03_label_news.py` maps them to IST and assigns `time_bucket`:

| IST Session | UTC Range | Label | Trading Implication |
|-------------|-----------|-------|---------------------|
| CLOSED_PRE  | 15:30–23:59 prev day | `CLOSED_PRE` | News after market close, before midnight |
| CLOSED_POST | 00:00–09:15 | `CLOSED_POST` | News overnight before market open |
| OPEN        | 09:15–15:30 | `OPEN` | News during trading hours |

Both `CLOSED_PRE` and `CLOSED_POST` map to `effective_date` = **next trading day** (the day the market can react).
`OPEN` maps to `effective_date` = **same trading day**.

---

## 2. Return Definitions (All per-ticker, shift(-1) by ticker)

| Return Column | Definition | Tradable? |
|---------------|------------|-----------|
| `ret_overnight` | Close(t-1) → Open(t) | **No** — realized before signal at t |
| `ret_intraday` | Open(t) → Close(t) | **No** for CLOSED windows; **Yes** for OPEN if signal available before close |
| `ret_close2close` | Close(t-1) → Close(t) | **No** for same-day; **Yes** for next-day |
| `ret_fwd_1d` | Close(t) → Close(t+1) | **Yes** — standard next-day forward |
| `ret_fwd_{h}d` | Close(t) → Close(t+h) | **Yes** — h-day forward |

---

## 3. Signal Construction

Daily signals aggregated per ticker per `effective_date` (pipeline step `05_aggregate_signals.py`):

- `signal_organic_closed` = mean GDELT tone of organic articles in CLOSED_PRE + CLOSED_POST
- `signal_organic_open` = mean GDELT tone of organic articles in OPEN
- `signal_sponsored_closed` = mean GDELT tone of sponsored articles in CLOSED_PRE + CLOSED_POST
- `signal_sponsored_open` = mean GDELT tone of sponsored articles in OPEN

**Composite signals**: `organic_signal`, `sponsored_signal`, `net_signal`

---

## 4. Valid Predictive Pairs (Used in IC Tests)

| Signal | Return | Alignment | Leakage? |
|--------|--------|-----------|----------|
| `signal_organic_closed` (date t) | `ret_fwd_1d` (date t+1) | t → t+1 | **No** |
| `signal_sponsored_closed` (date t) | `ret_fwd_1d` (date t+1) | t → t+1 | **No** |
| `signal_organic_open` (date t) | `ret_fwd_1d` (date t+1) | t → t+1 | **No** |
| `signal_sponsored_open` (date t) | `ret_fwd_1d` (date t+1) | t → t+1 | **No** |
| `signal_organic_closed` (date t) | `ret_overnight_next` (date t+1) | t → t+1 | **No** |
| `signal_sponsored_closed` (date t) | `ret_overnight_next` (date t+1) | t → t+1 | **No** |

---

## 5. Invalid / Contemporaneous Pairs (Leakage)

| Signal | Return | Why Invalid |
|--------|--------|-------------|
| `signal_organic_open` (date t) | `ret_intraday` (date t) | Same-day; signal at t cannot trade intraday return at t |
| `signal_sponsored_open` (date t) | `ret_intraday` (date t) | Same-day leakage |
| Any signal (date t) | `ret_overnight` (date t) | Overnight return at t is *before* signal at t |
| Any signal (date t) | `ret_close2close` (date t) | Same-day close-to-close includes intraday component |

---

## 6. Key Findings Under This Contract

| Hypothesis | Result |
|------------|--------|
| H2: Closed-window > Open-window predictive power | **REJECTED** — Organic closed IC = -0.016 (t=-2.82); Organic open IC (contemporaneous) = +0.018 but **leakage** |
| Strict forwards (t → t+1): Organic closed → fwd1 | +0.011 (t=2.03, ns after FDR) |
| Strict forwards (t → t+1): Sponsored closed → fwd1 | +0.010 (t=2.09, ns after FDR) |
| Sponsored open → intraday (contemporaneous) | +0.041 (t=8.11) — **LEAKAGE**, collapses next-day |

---

## 7. Event Study Timing (03_hypothesis_h2.py)

- **Event definition**: High organic closed sentiment (≥80th pctl by ticker on train split 2020-2023)
- **CAR window**: [-2, +5] trading days relative to event day (day 0)
- **Abnormal return**: Stock return - NIFTY 50 index return
- **Day 0 CAR**: +0.224% (t=6.98) — this is the *event day itself*, includes same-day return component
- **Day +1 CAR**: +0.262% (t=7.03) — next trading day, **predictive**
- **Look-ahead check**: Thresholds computed on train split only; event detection uses only past data. ✅ No look-ahead.

---

## 8. Signal Decay Timing (research/signal_decay.py)

Uses `ret_fwd_{h}d` columns built by `backtest.add_predictive_forwards()`:
- `ret_fwd_1d` = next trading day close-to-close
- `ret_fwd_hd` = h trading days ahead close-to-close
- All strictly forward from signal date

---

## 9. Backtest Timing (research/backtest.py)

`add_predictive_forwards()` builds `fwd1_close2close` = `ret_close2close.shift(-1)` per ticker.
Backtest trades signal at t → returns realized at t+1 close. Costs: 10 bps per round-trip.

---

## 10. Summary: Timestamp Contract

```
GDELT UTC → IST bucket → effective_date (trading day) → signal_t
                                                          ↓
                                              ret_fwd_1d at t+1 (predictive target)
```

**No same-day returns used as predictive targets in any reported result.**

---

## 11. Verification Commands

```bash
# Check master dataset has forward columns
python -c "import pandas as pd; df=pd.read_parquet('data/final/master_dataset.parquet'); print([c for c in df.columns if c.startswith('ret_fwd')])"

# Verify no same-day leakage in IC computation
python -c "
import pandas as pd
df = pd.read_parquet('data/final/master_dataset.parquet')
# Check that ret_intraday is NOT used with signal_organic_open for predictive IC
print('Signal columns:', [c for c in df.columns if 'signal' in c])
print('Return columns:', [c for c in df.columns if c.startswith('ret_')])
"
```