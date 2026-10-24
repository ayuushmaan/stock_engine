"""Tests for quantitative research modules: baselines, signal decay, backtest, and cross-market."""
import numpy as np
import pandas as pd
import pytest

from research.baselines import compute_spearman_ic, newey_west_tstat, benjamini_hochberg_fdr, evaluate_five_baselines
from research.signal_decay import compute_decay_curve
from research.backtest import run_factor_backtest
from research.cross_market import simulate_cross_market_comparison


def test_benjamini_hochberg_fdr():
    """Verify FDR adjustment maintains monotonicity and bounds."""
    p_vals = [0.001, 0.01, 0.04, 0.05, 0.20]
    adj_p = benjamini_hochberg_fdr(p_vals)
    assert len(adj_p) == len(p_vals)
    assert all(0.0 <= p <= 1.0 for p in adj_p)
    assert adj_p[0] <= adj_p[-1]


def test_newey_west_tstat():
    """Verify Newey-West t-statistic on synthetic stationary series."""
    np.random.seed(42)
    s = pd.Series(np.random.normal(0.05, 0.1, 100))
    t_stat, p_val = newey_west_tstat(s, lags=3)
    assert not np.isnan(t_stat)
    assert 0.0 <= p_val <= 1.0


def test_evaluate_five_baselines():
    """Verify all 5 baselines are computed and formatted properly."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "effective_date": pd.date_range("2023-01-01", periods=20, freq="B").repeat(10),
        "raw_tone": np.random.normal(0, 2, n),
        "finbert_sentiment": np.random.normal(0, 1, n),
        "sponsored_prob": np.random.uniform(0, 1, n),
        "time_bucket": np.random.choice(["OPEN", "CLOSED_PRE"], n),
        "ret_fwd_1d": np.random.normal(0, 0.02, n),
    })
    res = evaluate_five_baselines(df)
    assert len(res) == 5
    assert "5. Full Proposed Signal" in res
    assert "icir" in res["5. Full Proposed Signal"]
    assert "p_value_fdr" in res["5. Full Proposed Signal"]


def test_run_factor_backtest():
    """Verify factor backtest runs and outputs proper metrics and curves."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "effective_date": pd.date_range("2023-01-01", periods=20, freq="B").repeat(10),
        "ticker": [f"STOCK_{i%10}" for i in range(n)],
        "pred_score": np.random.normal(0, 1, n),
        "ret_fwd_1d": np.random.normal(0.001, 0.015, n),
    })
    perf_df, metrics = run_factor_backtest(df, cost_bps=10.0)
    assert len(perf_df) > 0
    assert metrics.annualized_volatility >= 0.0
    assert "cum_net" in perf_df.columns


def test_cross_market_simulation():
    """Verify cross market simulation payload."""
    data = simulate_cross_market_comparison()
    assert "NIFTY 50 (India)" in data
    assert "S&P 500 (US)" in data
    assert data["NIFTY 50 (India)"]["temporal_multiplier"] > 1.0
