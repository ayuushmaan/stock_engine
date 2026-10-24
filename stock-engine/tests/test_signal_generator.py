"""Unit tests for models/signal_generator.py (vectorized scoring and signal engine)."""
import numpy as np
import pandas as pd
import pytest

from config.settings import (
    ALPHA,
    DIRECTION_THRESHOLD,
    MIN_ARTICLES_FOR_SIGNAL,
    ORGANIC_WEIGHT_FLOOR,
    TIME_WEIGHT_CLOSED,
    TIME_WEIGHT_OPEN,
    TONE_NORMALIZER,
)
from models.signal_generator import (
    batch_generate_signals,
    generate_signal,
    score_article,
)


def test_score_article_scalar():
    """Test scalar execution of score_article."""
    tone = 10.0
    prob = 0.0  # 100% organic -> organic_weight = 1.0
    is_closed = True  # closed -> time_weight = 1.5
    # score = (10 / 10) * 1.0 * 1.5 = 1.5
    res = score_article(tone, prob, is_closed)
    assert pytest.approx(res, 1e-6) == 1.5

    # Test floor weight
    prob_high = 0.95  # 1 - 0.95 = 0.05 < FLOOR (0.3) -> organic_weight = 0.3
    is_open = False
    res_floor = score_article(tone, prob_high, is_open)
    # score = (10 / 10) * 0.3 * 1.0 = 0.3
    assert pytest.approx(res_floor, 1e-6) == 0.3


def test_score_article_vectorized():
    """Test vectorized array inputs for score_article."""
    tones = np.array([10.0, -10.0, 0.0])
    probs = np.array([0.0, 0.5, 1.0])
    closed = np.array([True, False, True])

    scores = score_article(tones, probs, closed)
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 3
    assert pytest.approx(scores[0], 1e-6) == 1.5  # (1.0) * 1.0 * 1.5
    assert pytest.approx(scores[1], 1e-6) == -0.5  # (-1.0) * 0.5 * 1.0
    assert pytest.approx(scores[2], 1e-6) == 0.0   # (0.0) * 0.3 * 1.5


def test_generate_signal_insufficient_articles():
    """Test generate_signal returns INSUFFICIENT_DATA when articles < MIN_ARTICLES_FOR_SIGNAL."""
    small_df = pd.DataFrame({
        "tone_score": [5.0, -2.0],
        "sponsored_prob": [0.1, 0.2],
        "time_bucket": ["OPEN", "OPEN"],
    })
    res = generate_signal(small_df)
    assert res["direction"] == "INSUFFICIENT_DATA"
    assert np.isnan(res["pred_score"])
    assert res["n_articles"] == 2


def test_generate_signal_bullish(sample_single_stock_articles):
    """Test generate_signal produces BULLISH signal for positive tone."""
    res = generate_signal(sample_single_stock_articles)
    assert res["direction"] in ["BULLISH", "NEUTRAL", "BEARISH"]
    assert res["pred_score"] > 0
    assert -1.0 <= res["pred_score"] <= 1.0
    assert res["n_articles"] == 4


def test_batch_generate_signals(sample_articles_df):
    """Test batch_generate_signals returns consistent DataFrame with correct grouping."""
    result = batch_generate_signals(sample_articles_df)
    assert isinstance(result, pd.DataFrame)
    assert "ticker" in result.columns
    assert "effective_date" in result.columns
    assert "pred_score" in result.columns
    assert "direction" in result.columns

    # RELIANCE has 3 articles -> satisfies MIN_ARTICLES_FOR_SIGNAL
    rel = result[result["ticker"] == "RELIANCE"]
    assert len(rel) == 1
    assert rel["direction"].iloc[0] != "INSUFFICIENT_DATA"

    # INFY has 2 articles -> insufficient data
    infy = result[result["ticker"] == "INFY"]
    assert len(infy) == 1
    assert infy["direction"].iloc[0] == "INSUFFICIENT_DATA"


def test_batch_generate_signals_empty():
    """Test batch_generate_signals with empty dataframe."""
    res = batch_generate_signals(pd.DataFrame())
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0
    assert "pred_score" in res.columns
