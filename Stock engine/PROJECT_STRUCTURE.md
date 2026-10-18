# Project Structure — NIFTY 50 News-Signal Research Engine

**Purpose:** Rigorous, reproducible negative-result benchmark for news-driven alpha on NIFTY 50.
**Status:** Complete pipeline (01–15), paper draft, frozen judge protocol, circularity guard.

---

## 📁 Source Code (Committed)

### `config/` — Central Configuration
| File | Role |
|------|------|
| `settings.py` | All tunable params, date splits, paths, seeds, logging |
| `nifty50_tickers.py` | Ticker list, keyword maps, sector/cap-tier maps |
| `source_tiers.py` | Domain quality tiers (Tier-1/2/3), PR wire lists, organic domains |
| `__init__.py` | Package marker |

### `pipeline/` — Numbered, Sequential Steps (01–15)
| Step | File | Description |
|------|------|-------------|
| **01** | `01_fetch_prices.py` | Download NIFTY 50 OHLCV from Yahoo Finance → compute overnight/intraday/close2close returns |
| **02** | `02_fetch_gdelt.py` | Query GDELT 2.0 GKG via BigQuery for NIFTY 50 entities → monthly Parquet chunks |
| **03** | `03_label_news.py` | Assign time buckets (CLOSED_POST/PRE/OPEN), effective dates, weak sponsored labels |
| **04** | `04_sponsored_classifier.py` | Train LightGBM on weak labels → score all articles with `sponsored_prob` |
| **04b** | `04_sponsored_scoring_engine.py` | Alternative HHEE scoring (deterministic gates + morphometrics) |
| **05** | `05_aggregate_signals.py` | Aggregate article scores → daily stock-level signals (organic/sponsored/net + 2×2 cells) |
| **06** | `06_build_master.py` | Join signals + returns → `master_dataset.parquet` (point-in-time, no leakage) |
| **07** | `07_sample_for_hand_labels.py` | Stratified sample for 200 human annotations |
| **08** | `08_honest_eval.py` | Train/eval honest classifier (non-circular features only) |
| **09** | `09_llm_labeler.py` | LLM-as-judge (OpenAI-compatible API) for sponsored/organic classification |
| **10** | `10_human_validation.py` | Load 200 hand labels, compute inter-annotator agreement |
| **11** | `11_scrape_bodies.py` | Polite scraping of article bodies for text-based classification |
| **12** | `12_local_judge.py` | CPU LLM judge (smaller model, no GPU needed) |
| **13** | `13_gpu_judge.py` | **Frozen GPU judge** — Qwen3-32B fp16, SHA-256 rubric, kappa gate (0.70) |
| **14** | `14_audit_judge.py` | Audit judge vs human labels, append to `judge_audits.json` |
| **15** | `15_paper_trade.py` | Paper-trading diary — fake INR ledger, predict/score cycle |
| **Utility** | `check_circularity.py` | **AST circularity guard** — CI-gated, fails build on label leakage |
| **Utility** | `calibrate_thresholds.py` | Calibrate HHEE tau_low/tau_high on validation set |
| **Utility** | `annotate_cli.py` | CLI for manual annotation workflow |
| **Core** | `scoring_engine.py` | HHEE 3-layer classifier (gates → evidence → thresholds) |
| **Core** | `llm_labeler.py` | LLM judge prompt schema, parsing, mock judge for tests |
| **Core** | `human_validation.py` | Gwet's AC1, Cohen's kappa, agreement metrics |
| **Core** | `_judge_parse.py` | Robust JSON parsing from LLM outputs |
| **Core** | `extractors/news_morphometrics.py` | Text features: promo/CTA/boilerplate/hype/risk/quote density |

### `models/` — Signal Generation & Trained Models
| File | Role |
|------|------|
| `signal_generator.py` | Vectorized signal engine: tone → weighted_score → tanh(α×raw) → BULLISH/BEARISH/NEUTRAL |
| `sponsored_classifier.pkl` | Trained LightGBM (weak-label supervised) |
| `honest_classifier.pkl` | Trained honest model (non-circular features only) |
| `__init__.py` | Package marker |

### `research/` — Hypothesis-Driven Analysis Scripts
| File | Hypothesis / Analysis |
|------|----------------------|
| `00_data_audit.ipynb` | Coverage, distributions, sanity checks |
| `01_classifier_validation.py/.ipynb` | Classifier AUC, PR curves, feature importance |
| `02_hypothesis_h1.py/.ipynb` | H1: Organic > Sponsored predictive power |
| `03_hypothesis_h2.py/.ipynb` | H2: Closed-window > Open-window predictive power |
| `04_2x2_interaction.py/.ipynb` | H1×H2: Organic-Closed is highest-alpha cell |
| `05_finbert_validation.py/.ipynb` | FinBERT sentiment vs GDELT tone benchmark |
| `06_baselines.py` | 5-baseline quant benchmark table |
| `07_signal_decay.py` | Signal horizon decay analysis |
| `08_backtest.py` | Predictive factor backtest (fwd1, fwd5, costs) |
| `09_cross_market.py` | Cross-market replication attempts |
| `10_propagation_dag.py` | Media cascade network analysis |
| `signal_decay.py` | Signal decay utilities |
| `propagation_dag.py` | Cascade DAG construction |
| `finbert_daily.py` | Daily FinBERT panel (sharded, resumable) |
| `baselines.py` | Baseline signal generators |
| `backtest.py` | Backtest engine (HAC t-stats, turnover, costs) |
| `cross_market.py` | Cross-market utilities |
| `benchmark_models.py` | Model comparison utilities |

### `tests/` — Unit Tests (22 pass)
| File | Coverage |
|------|----------|
| `test_signal_generator.py` | Vectorized scoring, edge cases, insufficient data |
| `test_scoring_engine.py` | HHEE classifier layers, attribution |
| `test_validation_pipeline.py` | LLM judge mock, Gwet's AC1, agreement metrics |
| `test_settings.py` | Config loading, seed determinism |
| `test_research_modules.py` | Research utilities |
| `conftest.py` | Pytest fixtures |

### `app/` — Dashboard
| File | Role |
|------|------|
| `app.py` | Streamlit dashboard: Signal Screener, Article Inspector, Factor Backtest, Paper Trading, Propagation Network, Model Validation |

### `paper/` — Publication
| File | Role |
|------|------|
| `PAPER_DRAFT.md` | Working draft: "Cheap Talk on the NSE..." — negative result, methodology contributions |

---

## 📁 Generated Artifacts (`.gitignore`d)

### `data/raw/`
- `prices/*.parquet` — 51 tickers × ~1500 days OHLCV
- `gdelt/gdelt_YYYY_MM.parquet` — 76 monthly GDELT chunks

### `data/processed/`
- `gdelt_labeled.parquet` — Articles with time_bucket, effective_date, weak_label
- `sponsored_scores.parquet` — All articles with `sponsored_prob` from LightGBM
- `returns.parquet` — Merged return panel with market regime
- `daily_signals.parquet` — Aggregated daily stock signals (2×2 cells)
- `finbert_shards_sponsored_scores/` — FinBERT article-level scores (sharded)

### `data/final/`
- `master_dataset.parquet` — Point-in-time joined signals + forwards
- `hand_labels.parquet` / `hand_labels_sample.parquet` / `hand_labels_workbook.csv` — 200 human annotations
- `llm_annotations.parquet/csv` — LLM judge outputs
- `local_judge_annotations.parquet` — CPU judge outputs
- `gpu_judge_annotations.parquet` — Frozen GPU judge outputs
- `finbert_scores.parquet` — FinBERT daily scores
- `article_bodies.parquet` — Scraped article texts
- `paper_predictions.parquet` — Paper trading diary predictions

### `outputs/tables/`
| File | Description |
|------|-------------|
| `circularity_audit.json` | AST audit results (CI gate) |
| `five_baselines_benchmark.json` | 5-baseline quant table (IC, t-stat, FDR) |
| `inter_annotator_agreement.json` | Human vs LLM vs Weak kappa/F1/AC1 |
| `honest_eval_report.json` | Honest classifier metrics (AUC 0.705) |
| `judge_audits.json` | Frozen judge audit trail |
| `hhee_v2_weights.json` | Learned HHEE morphometric weights |
| `factor_backtest_metrics.json` | Backtest: gross/net returns, Sharpe, DD, turnover |
| `paper_scoreboard.json` | Paper trading diary summary |
| `paper_equity.json` | Paper trading equity curve |
| `propagation_cascades.json` | Media cascade clusters |
| `calibration_results.json` | HHEE threshold calibration |
| `model_comparison.json` | Classifier comparisons |
| `h2_predictive.json` | H2 predictive results |
| `finbert_*.json` | FinBERT validation outputs |
| `cross_market_replication.json` | Cross-market results |
| `signal_decay_analysis.json` | Signal horizon decay |
| `scoring_summary.json` | HHEE score distributions |

### `outputs/figures/`
Organized by analysis:
- Root: `feature_importance.png`, `precision_recall.png`, `tone_distribution.png`, `backtest_equity_curve.png`, `propagation_dag_network.png`, `signal_decay_curve.png`, `scoring_distribution.png`, `pareto_calibration_frontier.png`, `cross_market_comparison.png`
- `h1/` — H1 organic vs sponsored violin, rolling IC
- `h2/` — H2 event study CAR
- `interaction/` — 2×2 regime grids, per-stock correlations, overnight contrarian, CAR decomposition
- `honest_eval/` — ROC, PR, feature importance, calibration
- `classifier_validation/` — Repro manifest, label distributions, feature importance audit, ambiguous articles, tone violins, top domains
- `finbert_validation/` — Distribution by bucket, FinBERT vs GDELT scatter
- `data_audit/` — Coverage heatmap, articles/month, return distribution

### `models/`
- `sponsored_classifier.pkl` — LightGBM (weak-label)
- `honest_classifier.pkl` — LogisticRegression (non-circular features)

---

## 🚀 Key Entry Points

| Goal | Command |
|------|---------|
| Full pipeline (dry-run) | `python pipeline/01_fetch_prices.py --dry-run && python pipeline/02_fetch_gdelt.py --dry-run && python pipeline/03_label_news.py --dry-run && python pipeline/04_sponsored_classifier.py --dry-run && python pipeline/05_aggregate_signals.py --dry-run` |
| Circularity check (CI) | `python pipeline/check_circularity.py` |
| Train frozen judge | `python pipeline/13_gpu_judge.py` |
| Audit judge | `python pipeline/14_audit_judge.py` |
| Paper trade cycle | `python pipeline/15_paper_trade.py predict && python pipeline/15_paper_trade.py score` |
| Run tests | `pytest tests/ -v` |
| Launch dashboard | `streamlit run app/app.py` |
| Research: 5-baseline table | `python research/06_baselines.py` |
| Research: backtest | `python research/08_backtest.py` |

---

## 🔒 Reproducibility Guarantees

1. **Deterministic seeds** — `seed_everything()` called at top of every pipeline script
2. **AST circularity guard** — `check_circularity.py` fails CI if label leakage detected
3. **Frozen judge protocol** — `JUDGE_SPEC` with SHA-256 rubric hash, kappa ≥ 0.70 gate
4. **No synthetic fallbacks** — Scripts fail loudly on missing data
5. **Point-in-time joins** — `effective_date` mapping, no look-ahead in `master_dataset`
6. **Versioned artifacts** — Parquet schemas, model pickles, JSON tables with timestamps

---

## 📊 Current Results Summary (from `outputs/tables/`)

| Metric | Value | Significance |
|--------|-------|--------------|
| Honest classifier CV AUC | 0.705 | Metadata-only, non-circular |
| HHEE-v2 text-only CV AUC | 0.688 | Learned on 149 scraped bodies |
| Full HHEE + Layer-0 gates | 0.958 | Circular (PR wire 82.8% vs 0%) |
| Qwen3-32B vs Human κ | 0.75 | F1 0.81 (n=129 excl. UNCLEAR) |
| net_signal → fwd1 IC | +0.005 | t=1.2, ns |
| organic → fwd1 IC | +0.008 | t=1.9, ns |
| FinBERT daily → fwd1 IC | +0.008 | t=1.9, ns |
| Q5–Q1 Gross Ann. Return | +11.2% | Sharpe 0.8 |
| Q5–Q1 Net Ann. Return | −22.4% | Sharpe −1.6, 66.7% turnover |
| Buy-and-hold (NIFTY 50) | +22.9%/yr | Hurdle |

**Conclusion:** GDELT-tone news carries no actionable next-day alpha for NIFTY 50. The contribution is a fully reproducible negative-result benchmark with leakage diagnostics, frozen-judge protocol, and paper-trading diary.