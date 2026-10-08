"""NIFTY 50 Quantitative News Intelligence & Signal Research Dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(
    page_title="NIFTY 50 News Intelligence & Signal Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 12px;
    }
    .badge-organic {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-sponsored {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-unclear {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("NIFTY 50 Source Credibility & Temporal Asymmetry Engine")
st.caption("Quantitative News Analysis, Content Classification & Factor Research")

# Sidebar navigation
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio(
    "Select View",
    [
        "Signal Screener",
        "Article Inspector",
        "Factor Backtest",
        "Propagation Network",
        "Model Validation",
    ],
)

# -------------------------------------------------------------
# TAB 1: Alpha Signal Screener
# -------------------------------------------------------------
if app_mode == "Signal Screener":
    st.subheader("NIFTY 50 Cross-Sectional Alpha Signals — REAL DATA")
    st.markdown("Daily signals from `data/processed/daily_signals.parquet` (fallback: `data/final/master_dataset.parquet`). No synthetic values.")
    from pathlib import Path as _Path
    _bench_path = _Path("outputs/tables/five_baselines_benchmark.json")
    if _bench_path.exists():
        _bench = json.load(open(_bench_path, encoding="utf-8"))
        _full = _bench.get("5. Full Proposed Signal", {})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Universe", "50 Constituents", "NSE Benchmark")
        with col2:
            st.metric("Full-signal Mean IC (on-disk)", f"{_full.get('mean_ic', 0):+.4f}", f"HAC t={_full.get('hac_tstat', 0)}")
        with col3:
            st.metric("Full-signal Ann. L/S", f"{_full.get('ann_ls_return', 0):+.1f}%", f"FDR p={_full.get('p_value_fdr', 1)}")
        with col4:
            st.metric("Status", "NULL RESULT", "IC ~0.004, ns — repair in progress")
    # Load real signals: latest date available
    _sig_path = _Path("data/processed/daily_signals.parquet")
    _master_path = _Path("data/final/master_dataset.parquet")
    try:
        if _sig_path.exists():
            _df = pd.read_parquet(_sig_path)
            _date_col = "effective_date" if "effective_date" in _df.columns else "date"
            _df[_date_col] = pd.to_datetime(_df[_date_col])
            _latest = _df[_date_col].max()
            df_screener = _df[_df[_date_col] == _latest].copy()
            st.caption(f"Showing {_latest.date()} — {len(df_screener)} tickers with signals (latest in daily_signals.parquet).")
            # Derive alpha score + direction from net_signal if pred_score absent
            if "pred_score" not in df_screener.columns:
                df_screener["Alpha Score"] = df_screener.get("net_signal", 0).fillna(0).round(3)
            else:
                df_screener["Alpha Score"] = df_screener["pred_score"].round(3)
            def _dir(s):
                return "BULLISH" if s > 0.10 else ("BEARISH" if s < -0.10 else "NEUTRAL")
            df_screener["Signal Direction"] = df_screener["Alpha Score"].apply(_dir)
            _show = [c for c in ["ticker", "Alpha Score", "Signal Direction", "article_count_total", "article_count_organic", "article_count_sponsored", "signal_organic_closed", "signal_organic_open"] if c in df_screener.columns]
            st.dataframe(df_screener[_show].sort_values("Alpha Score", ascending=False), use_container_width=True, hide_index=True)
        elif _master_path.exists():
            st.warning("daily_signals.parquet missing — showing latest master_dataset.parquet snapshot.")
            _df = pd.read_parquet(_master_path)
            _date_col = "date" if "date" in _df.columns else "effective_date"
            _latest = pd.to_datetime(_df[_date_col]).max()
            st.caption(f"Showing {_latest.date()} from master_dataset.")
            st.dataframe(_df[pd.to_datetime(_df[_date_col]) == _latest].sort_values("net_signal", ascending=False).head(50), use_container_width=True, hide_index=True)
        else:
            st.error("No signal data found. Run pipeline/05_aggregate_signals.py first. Refusing to show synthetic data.")
    except Exception as e:
        st.error(f"Failed to load real signals: {e}")

# -------------------------------------------------------------
# TAB 2: Article Inspector
# -------------------------------------------------------------
elif app_mode == "Article Inspector":
    st.subheader("Article Content & Morphometrics Inspector")
    st.markdown("Decomposes news releases through the 3-Layer Hierarchical Hybrid Evidence Engine (HHEE).")

    from pipeline.scoring_engine import score_single_article

    sample_choice = st.selectbox(
        "Choose a Test Case or Enter Custom Content",
        [
            "Custom Input",
            "PR Wire Example (BusinessWire Advertorial)",
            "Organic Journalism (RBI Rate Policy - Reuters)",
            "Corporate Milestone Release",
        ]
    )

    if sample_choice == "PR Wire Example (BusinessWire Advertorial)":
        init_title = "Global Leader Announces Milestone and Award in Enterprise Cloud Services"
        init_body = "Click here to buy our enterprise solution and subscribe to our newsletter for free demo access. We are pleased to announce product expansion."
        init_domain = "businesswire.com"
        init_url = "https://businesswire.com/news/123"
    elif sample_choice == "Organic Journalism (RBI Rate Policy - Reuters)":
        init_title = "Reserve Bank of India holds repo rate steady at 6.5% amid persistent inflation risks"
        init_body = "The monetary policy committee voted 5-1 to maintain status quo as consumer food inflation remained elevated, analysts from Nomura and Goldman noted."
        init_domain = "reuters.com"
        init_url = "https://reuters.com/markets/rbi-rates"
    elif sample_choice == "Corporate Milestone Release":
        init_title = "Company reports quarterly expansion and operational progress"
        init_body = "We are pleased to report strong performance across key domestic operating segments following recent strategic capital investments."
        init_domain = "dailyexpressnews.in"
        init_url = "https://dailyexpressnews.in/post/456"
    else:
        init_title = ""
        init_body = ""
        init_domain = ""
        init_url = ""

    col_in1, col_in2 = st.columns([1, 1])
    with col_in1:
        headline = st.text_input("Article Headline", value=init_title)
        domain = st.text_input("Source Domain", value=init_domain)
        url = st.text_input("Source URL", value=init_url)
    with col_in2:
        text = st.text_area("Article Body Content", value=init_body, height=130)

    if st.button("Classify Content", type="primary"):
        res = score_single_article(headline=headline, text=text, domain=domain, url=url)

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Sponsored Probability S(A)", f"{res['score']:.4f}")
        with res_col2:
            cls = res["classification"].upper()
            st.metric("Classification", cls)
        with res_col3:
            st.metric("Trigger Mechanism", res.get("layer_triggered", "Layer 1 Score"))

        st.markdown("### Feature Attribution Breakdown")
        attr = res.get("attribution", {})
        df_attr = pd.DataFrame([{"Feature": k, "Weight * Value": v} for k, v in attr.items()])
        st.dataframe(df_attr, use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 3: Factor Backtest
# -------------------------------------------------------------
elif app_mode == "Factor Backtest":
    st.subheader("Quantitative Factor Portfolio Simulation — REAL BACKTEST")
    st.caption("Uses `data/final/master_dataset.parquet`: signal=`net_signal`, return=`ret_close2close`. Same-day (contemporaneous) — predictive fwd version in progress.")

    cost_slippage = st.slider("Transaction Cost / Slippage per trade (bps)", min_value=0, max_value=50, value=10, step=5)

    from research.backtest import run_factor_backtest
    from pathlib import Path as _P
    _mp = _P("data/final/master_dataset.parquet")
    if not _mp.exists():
        st.error("master_dataset.parquet missing. Run pipeline/06_build_master.py first. Refusing synthetic backtest.")
        st.stop()
    df_bt = pd.read_parquet(_mp)
    # backtest engine expects effective_date col; master uses date
    if "effective_date" not in df_bt.columns and "date" in df_bt.columns:
        df_bt = df_bt.rename(columns={"date": "effective_date"})
    _sig = "net_signal" if "net_signal" in df_bt.columns else "pred_score"
    _ret = "ret_close2close" if "ret_close2close" in df_bt.columns else "ret_fwd_1d"

    perf_df, metrics = run_factor_backtest(df_bt, signal_col=_sig, return_col=_ret, cost_bps=float(cost_slippage))

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Annualized Gross Return", f"{metrics.annualized_return:.1f}%")
    with m_col2:
        st.metric("Net Annualized Return", f"{metrics.net_annualized_return:.1f}%")
    with m_col3:
        st.metric("Net Sharpe Ratio", f"{metrics.net_sharpe_ratio:.2f}")
    with m_col4:
        st.metric("Daily Turnover", f"{metrics.daily_turnover:.1f}%")

    st.image("outputs/figures/backtest_equity_curve.png", caption="Cumulative Wealth & Drawdown Curves")

# -------------------------------------------------------------
# TAB 4: Propagation Network
# -------------------------------------------------------------
elif app_mode == "Propagation Network":
    st.subheader("Cross-Media News Cascading & Propagation Network")
    st.markdown("Traces how PR releases cascade through aggregators to national newsrooms.")
    st.image("outputs/figures/propagation_dag_network.png", caption="Information Cascading Graph across Hop Levels")

    with open("outputs/tables/propagation_cascades.json", "r", encoding="utf-8") as f:
        cascades = json.load(f)
    st.write(f"**Indexed Information Cascades**: {len(cascades)} event clusters")
    st.dataframe(pd.DataFrame(cascades).drop(columns=["nodes"]), use_container_width=True)

# -------------------------------------------------------------
# TAB 5: Model Validation
# -------------------------------------------------------------
elif app_mode == "Model Validation":
    st.subheader("Non-Circularity Verification & Multi-Judge Agreement")

    st.markdown("""
    ### 1. Structural Non-Circularity Verification (AST Guard)
    - Verifies that model input features do not overlap with weak-label heuristics.
    - Static AST analysis confirms zero contaminated features in the final training dataset.
    """)

    st.markdown("### 2. Multi-Judge Inter-Annotator Agreement (200 Benchmark Annotations)")
    with open("outputs/tables/inter_annotator_agreement.json", "r", encoding="utf-8") as f:
        agreement = json.load(f)
    st.json(agreement)

    st.markdown("""
    ### 3. Five-Baseline Quant Benchmark Table
    """)
    with open("outputs/tables/five_baselines_benchmark.json", "r", encoding="utf-8") as f:
        baselines = json.load(f)
    df_b = pd.DataFrame.from_dict(baselines, orient="index")
    st.dataframe(df_b, use_container_width=True)
