# Architecture Diagram

```mermaid
flowchart TD
    %% Data Sources
    subgraph DataSources [Data Sources]
        YF[(Yahoo Finance\nNIFTY 50 OHLCV)]
        BQ[(BigQuery GDELT 2.0 GKG\n1.4M news records)]
    end

    %% Pipeline: Data Ingestion
    subgraph Pipeline [Pipeline Steps 01-06]
        P1[01_fetch_prices.py\n→ data/raw/prices/]
        P2[02_fetch_gdelt.py\n→ data/raw/gdelt/]
        P3[03_label_news.py\nIST bucketing, weak labels\n→ data/processed/gdelt_labeled.parquet]
        P4[04_sponsored_classifier.py\nLightGBM on weak labels\n→ data/processed/sponsored_scores.parquet]
        P5[05_aggregate_signals.py\nDaily ticker signals\n→ data/processed/daily_signals.parquet]
        P6[06_build_master.py\nMerge signals + returns\n→ data/final/master_dataset.parquet]
    end

    %% Pipeline: Labeling & Validation
    subgraph Labeling [Pipeline Steps 07-15]
        P7[07_sample_for_hand_labels.py\n200 stratified articles]
        P8[08_honest_eval.py\nNon-circular features only\nAUC 0.705]
        P9[09_llm_labeler.py\nAPI LLM judge]
        P10[10_human_validation.py\nInter-annotator agreement\nκ=0.75 vs Qwen3-32B]
        P11[11_scrape_bodies.py\nPolite scraping → article bodies]
        P12[12_local_judge.py\nCPU NLI judge]
        P13[13_gpu_judge.py\nFrozen Qwen3-32B fp16\nSHA-256 rubric, κ≥0.70 gate]
        P14[14_audit_judge.py\nAudit trail → judge_audits.json]
        P15[15_paper_trade.py\nFake INR diary]
    end

    %% Core Modules
    subgraph Core [Core Modules]
        CE[check_circularity.py\nAST circularity guard]
        SE[scoring_engine.py\nHHEE 3-layer classifier]
        LL[llm_labeler.py\nLLM judge schema/parsing]
        HV[human_validation.py\nGwet's AC1, Cohen's κ]
        SG[signal_generator.py\nVectorized cross-sectional signals]
    end

    %% Research
    subgraph Research [Research Analyses]
        R1[06_baselines.py\n5-baseline benchmark]
        R2[08_backtest.py\nPredictive factor backtest]
        R3[07_signal_decay.py\nHorizon decay T+1..T+20]
        R4[robustness_tests.py\nPlacebo, permutation, FDR, sweep]
        R5[propagation_dag.py\nMeasured cascades + bootstrap CI]
    end

    %% Outputs
    subgraph Outputs [Outputs]
        OT[(outputs/tables/*.json\nMeasured results only)]
        OF[(outputs/figures/*.png\nCharts)]
        MT[(models/*.pkl\nTrained classifiers)]
    end

    %% Flows
    YF --> P1
    BQ --> P2
    P1 --> P6
    P2 --> P3
    P3 --> P4
    P3 --> P5
    P4 --> P5
    P5 --> P6
    
    P6 --> P7
    P7 --> P8
    P7 --> P9
    P7 --> P10
    P11 --> P12
    P11 --> P13
    P13 --> P14
    P6 --> P15
    
    P4 --> CE
    P8 --> CE
    CE -.->|CI Gate| OT
    
    P1 --> SG
    P5 --> SG
    P6 --> R1
    P6 --> R2
    P6 --> R3
    P6 --> R4
    P3 --> R5
    
    P1 --> OT
    P2 --> OT
    P3 --> OT
    P4 --> OT
    P5 --> OT
    P6 --> OT
    P8 --> OT
    P10 --> OT
    P13 --> OT
    P14 --> OT
    P15 --> OT
    R1 --> OT
    R2 --> OT
    R3 --> OT
    R4 --> OT
    R5 --> OT
    
    R1 --> OF
    R2 --> OF
    R3 --> OF
    R5 --> OF
    
    P4 --> MT
    P8 --> MT
```

---

## Plain-English Summary (for non-experts)

**What this project does:**
We tested whether news articles about NIFTY 50 companies can predict stock returns the next day. We looked at two ideas:
1. **Source matters**: "Real" journalism (editorial) might be more informative than corporate press releases (sponsored/PR).
2. **Timing matters**: News published while the market is closed (overnight) might be more valuable because it hasn't been priced in yet.

**What we found (the honest answer):**
**Nothing works.** After rigorous testing with real data, proper forward-looking evaluation (no peeking at same-day returns), and statistical corrections for multiple testing — **none of the signals produce statistically significant predictive power.**

**Key numbers:**
- Best signal (organic news in closed market hours): IC ≈ +0.011 (t ≈ 2.0, not significant after multiple-testing correction)
- A long-short portfolio based on this would lose **−22% per year** after 10 bps transaction costs (66% daily turnover)
- Buy-and-hold NIFTY 50 returned **+23%/year** — the hurdle we couldn't beat
- Sponsored content *is* detectable (AI judge agrees with humans at κ=0.75), but **not monetizable** with tone-level signals

**Why this matters:**
- Many papers claim news sentiment predicts returns. We show the apparent signal is usually **leakage** (using same-day returns that couldn't actually be traded on).
- Our "circularity guard" (automated code audit) proves our honest classifier doesn't cheat by using the labels it's trying to predict.
- The frozen LLM judge protocol ensures human-AI agreement metrics can't be gamed.
- The paper-trading diary keeps future claims honest.

**Limitations we're upfront about:**
- 96.5% of "headlines" are just URL slugs (no real text)
- Only 200 hand-labeled articles (34 sponsored)
- Current NIFTY 50 constituents only (survivorship bias)
- Business-day calendar, not actual NSE holidays
- No US cross-market data (simulated placeholders quarantined)

**Bottom line:** Indian equity PR is detectable but not tradeable with tone-level signals. The contribution is a **fully reproducible negative-result benchmark** with leakage diagnostics that keep the field honest.