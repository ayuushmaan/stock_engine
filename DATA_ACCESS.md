# Data Access Guide — NIFTY 50 News-Signal Research Engine

This document describes how to access and reproduce all data used in this project.

---

## 1. Price Data (NIFTY 50 OHLCV)

**Source:** Yahoo Finance via `yfinance`  
**Tickers:** 50 current NIFTY 50 constituents + `^NSEI` (index)  
**Date Range:** 2019-12-01 to 2026-03-31 (configurable in `config/settings.py`)  
**Frequency:** Daily  
**Fields:** Open, High, Low, Close, Volume  
**Derived:** `ret_overnight`, `ret_intraday`, `ret_close2close`, `market_regime`

**Access:**
```bash
# Dry-run (3 tickers, ~50 days)
python pipeline/01_fetch_prices.py --dry-run

# Full run (all 50 tickers, 6+ years)
python pipeline/01_fetch_prices.py
```

**Output:** `data/raw/prices/{TICKER}.parquet` + `data/processed/returns.parquet`

**Notes:**
- Yahoo Finance has rate limits; script includes retries + 3s delays
- `auto_adjust=True` handles splits/dividends
- Market regime computed from NIFTY 50 60-day rolling return (BULL >10%, BEAR <-10%, else SIDEWAYS)
- NSE holidays approximated as business days; replace `_get_nse_trading_days()` with actual calendar for production

---

## 2. News Data (GDELT 2.0 GKG)

**Source:** Google BigQuery — `gdelt-bq.gdeltv2.gkg_partitioned`  
**Date Range:** 2020-01 to 2026-03 (monthly partitions)  
**Filter:** Indian financial news domains (see `INDIA_DOMAIN_PATTERN` in `pipeline/02_fetch_gdelt.py`)  
**Entity Matching:** NIFTY 50 company keywords via `V2Organizations` + URL matching  
**Fields:** `DATE`, `SourceCommonName`, `DocumentIdentifier`, `V2Tone`, `V2Organizations`

**Access Requirements:**
1. Google Cloud Project with BigQuery API enabled
2. Set `GOOGLE_CLOUD_PROJECT` env var or edit `config/settings.py:GCP_PROJECT_ID`
3. Billing enabled (queries ~30 GB/month max, controlled by `MAX_BYTES_BILLED`)

**Run:**
```bash
# Dry-run (Jan 2024, 5 companies)
python pipeline/02_fetch_gdelt.py --dry-run

# Full run (all months, all companies)
python pipeline/02_fetch_gdelt.py --start-month 2020-01 --end-month 2026-03

# Single month
python pipeline/02_fetch_gdelt.py --month 2024-01
```

**Output:** `data/raw/gdelt/gdelt_YYYY_MM.parquet` (76 monthly files)

**Cost Estimate:** ~$15-25 total for full 2020-2026 fetch at on-demand pricing

---

## 3. Article Bodies (Full Text)

**Source:** Polite scraping of `DocumentIdentifier` URLs  
**Tool:** `pipeline/11_scrape_bodies.py` (respects `robots.txt`, 2s delay, caches HTML)  
**Coverage:** ~149/200 hand-labeled articles + subset for FinBERT

**Run:**
```bash
# Scrape bodies for hand-labeled subset
python pipeline/11_scrape_bodies.py

# Scrape for FinBERT panel (sharded, resumable)
python research/finbert_daily.py
```

**Output:** `data/final/article_bodies.parquet`, `data/processed/finbert_shards_sponsored_scores/`

---

## 4. Human Labels (200 Articles)

**Source:** Manual annotation of stratified sample  
**Script:** `pipeline/07_sample_for_hand_labels.py` → `pipeline/10_human_validation.py`  
**Distribution:** 34 sponsored / 166 organic  
**Format:** `data/final/hand_labels.parquet` + `hand_labels_workbook.csv` (for annotation)

**Inter-annotator Agreement:** See `outputs/tables/inter_annotator_agreement.json`

---

## 5. LLM Judge Annotations

| Judge | Model | Config | Output |
|-------|-------|--------|--------|
| Local (CPU) | Qwen2.5-7B-Instruct 4-bit | `pipeline/12_local_judge.py` | `local_judge_annotations.parquet` |
| GPU (frozen) | Qwen3-32B fp16 | `pipeline/13_gpu_judge.py` | `gpu_judge_annotations.parquet` |
| API (OpenAI-compatible) | GPT-4o / etc. | `pipeline/09_llm_labeler.py` | `llm_annotations.parquet` |

**Frozen Judge Spec:** `pipeline/13_gpu_judge.py:JUDGE_SPEC` — SHA-256 rubric, kappa ≥ 0.70 gate

---

## 6. Model Artifacts

| Model | Training Data | Features | Output |
|-------|---------------|----------|--------|
| Sponsored Classifier | Weak labels (2020-2023) | 13 metadata features | `models/sponsored_classifier.pkl` |
| Honest Classifier | Weak labels (2020-2023) | 5 non-circular features | `models/honest_classifier.pkl` |
| HHEE v2 (text) | 149 scraped bodies | 6 morphometrics | `outputs/tables/hhee_v2_weights.json` |

---

## 7. Key Configurable Parameters

All in `config/settings.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FETCH_START` / `FETCH_END` | 2019-12-01 / 2026-03-31 | Price/GDELT fetch window |
| `TRAIN_START` / `TRAIN_END` | 2020-01-01 / 2023-12-31 | Signal/classifier train period |
| `VALIDATION_START` / `VALIDATION_END` | 2024-01-01 / 2024-06-30 | Validation period |
| `HOLDOUT_START` / `HOLDOUT_END` | 2024-07-01 / 2026-03-31 | Holdout/test period |
| `ALPHA` | 1.8 | tanh scaling for signal |
| `DIRECTION_THRESHOLD` | 0.10 | BULLISH/BEARISH threshold |
| `SPONSORED_PROB_HIGH` / `LOW` | 0.70 / 0.30 | HHEE classification thresholds |
| `MIN_ARTICLES_FOR_SIGNAL` | 3 | Minimum articles per stock-day |

---

## 8. Reproducing Key Results

| Result | Command | Output |
|--------|---------|--------|
| Circularity audit | `python pipeline/check_circularity.py` | `outputs/tables/circularity_audit.json` |
| 5-baseline table | `python research/06_baselines.py` | `outputs/tables/five_baselines_benchmark.json` |
| Factor backtest | `python research/08_backtest.py` | `outputs/tables/factor_backtest_metrics.json` |
| Judge audit | `python pipeline/14_audit_judge.py` | `outputs/tables/judge_audits.json` |
| Paper trade cycle | `python pipeline/15_paper_trade.py predict && python pipeline/15_paper_trade.py score` | `outputs/tables/paper_scoreboard.json` |

---

## 9. Compute Requirements

| Component | Requirement |
|-----------|-------------|
| CPU pipeline (01-05, 06, 08) | 8+ GB RAM, 4+ cores, ~30 min full run |
| GPU judge (13) | NVIDIA GPU 16+ GB VRAM (32B fp16) or 6+ GB (7B 4-bit) |
| FinBERT daily | GPU recommended, sharded for resumability |
| BigQuery | GCP project, ~$25 budget |

---

## 10. Directory Structure (Generated)

```
data/
├── raw/
│   ├── prices/           # 51 ticker parquets
│   └── gdelt/            # 76 monthly GDELT parquets
├── processed/
│   ├── returns.parquet
│   ├── gdelt_labeled.parquet
│   ├── sponsored_scores.parquet
│   └── daily_signals.parquet
└── final/
    ├── master_dataset.parquet
    ├── hand_labels.parquet
    ├── gpu_judge_annotations.parquet
    ├── article_bodies.parquet
    └── paper_predictions.parquet

outputs/
├── tables/               # All JSON result tables
└── figures/              # All PNG figures (organized by analysis)

models/
├── sponsored_classifier.pkl
└── honest_classifier.pkl
```

---

## 11. Troubleshooting

| Issue | Fix |
|-------|-----|
| BigQuery permission denied | Enable BigQuery API, check IAM roles (BigQuery User + Job User) |
| yfinance rate limited | Increase retry delay in `download_ticker()` |
| GPU OOM on judge | Use `--model Qwen/Qwen2.5-7B-Instruct --fp16` or 4-bit |
| Missing NSE holidays | Replace `_get_nse_trading_days()` with actual calendar CSV |
| Circularity check fails | Check `outputs/tables/circularity_audit.json` for contaminated features |

---

## 12. Citation

If you use this data/pipeline, please cite:

```
Ayushmaan (2026). Cheap Talk on the NSE: Source Credibility, Market Timing, 
and the Limits of News-Driven Return Prediction for NIFTY 50. 
SSRN Working Paper. https://github.com/ayuushmaan/stock_engine
```