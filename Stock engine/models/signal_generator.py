"""Signal Generator — converts article-level scores to final prediction signals.

Implements the signal formula from the research spec:

    1. article_score    = GDELT_tone / TONE_NORMALIZER  (≈ [-1, 1])
    2. organic_weight   = max(1 - sponsored_prob, ORGANIC_WEIGHT_FLOOR)
    3. time_weight      = TIME_WEIGHT_CLOSED if closed window, else TIME_WEIGHT_OPEN
    4. weighted_score   = article_score × organic_weight × time_weight
    5. raw_signal       = mean(weighted_scores for stock on effective_date)
    6. pred_score       = tanh(ALPHA × raw_signal)
    7. direction:
         pred_score > +THRESHOLD → BULLISH
         pred_score < -THRESHOLD → BEARISH
         else                    → NEUTRAL

All tunable parameters are sourced from config/settings.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    ALPHA,
    DIRECTION_THRESHOLD,
    MIN_ARTICLES_FOR_SIGNAL,
    ORGANIC_WEIGHT_FLOOR,
    SPONSORED_PROB_HIGH,
    SPONSORED_PROB_LOW,
    TIME_WEIGHT_CLOSED,
    TIME_WEIGHT_OPEN,
    TONE_NORMALIZER,
)


def score_article(
    tone: float | np.ndarray | pd.Series,
    sponsored_prob: float | np.ndarray | pd.Series,
    is_closed_window: bool | np.ndarray | pd.Series,
) -> float | np.ndarray:
    """Compute weighted score for article(s) (scalar or vectorized).

    Parameters
    ----------
    tone : float or array-like
        Raw GDELT tone score (typically [-20, +20]).
    sponsored_prob : float or array-like
        Probability that article is sponsored (0-1).
    is_closed_window : bool or array-like
        True if article was published during market-closed hours.

    Returns
    -------
    float or np.ndarray
        Weighted article score(s).
    """
    tone_arr = np.asarray(tone, dtype=float)
    prob_arr = np.asarray(sponsored_prob, dtype=float)
    closed_arr = np.asarray(is_closed_window, dtype=bool)

    article_score = tone_arr / TONE_NORMALIZER
    organic_weight = np.maximum(1.0 - prob_arr, ORGANIC_WEIGHT_FLOOR)
    time_weight = np.where(closed_arr, TIME_WEIGHT_CLOSED, TIME_WEIGHT_OPEN)
    res = article_score * organic_weight * time_weight
    return float(res) if np.ndim(res) == 0 else res


def generate_signal(
    articles: pd.DataFrame,
    tone_col: str = "tone_score",
    prob_col: str = "sponsored_prob",
    bucket_col: str = "time_bucket",
) -> dict:
    """Generate a single prediction signal from a group of articles.

    Parameters
    ----------
    articles : DataFrame
        Articles for one stock on one effective_date.
    tone_col, prob_col, bucket_col : str
        Column names.

    Returns
    -------
    dict with keys:
        raw_signal, pred_score, direction, n_articles,
        n_organic, n_sponsored
    """
    n_articles = len(articles)
    if n_articles < MIN_ARTICLES_FOR_SIGNAL:
        return {
            "raw_signal": np.nan,
            "pred_score": np.nan,
            "direction": "INSUFFICIENT_DATA",
            "n_articles": n_articles,
            "n_organic": 0,
            "n_sponsored": 0,
        }

    tones = articles[tone_col].fillna(0.0).to_numpy() if tone_col in articles else np.zeros(n_articles)
    probs = articles[prob_col].fillna(0.5).to_numpy() if prob_col in articles else np.full(n_articles, 0.5)
    buckets = articles[bucket_col].to_numpy() if bucket_col in articles else np.array(["OPEN"] * n_articles)
    is_closed = np.isin(buckets, ["CLOSED_POST", "CLOSED_PRE"])

    weighted_scores = score_article(tones, probs, is_closed)
    raw_signal = float(np.mean(weighted_scores))
    pred_score = float(np.tanh(ALPHA * raw_signal))

    if pred_score > DIRECTION_THRESHOLD:
        direction = "BULLISH"
    elif pred_score < -DIRECTION_THRESHOLD:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    n_organic = int((probs < SPONSORED_PROB_LOW).sum())
    n_sponsored = int((probs > SPONSORED_PROB_HIGH).sum())

    return {
        "raw_signal": raw_signal,
        "pred_score": pred_score,
        "direction": direction,
        "n_articles": n_articles,
        "n_organic": n_organic,
        "n_sponsored": n_sponsored,
    }


def batch_generate_signals(
    df: pd.DataFrame,
    ticker_col: str = "ticker",
    date_col: str = "effective_date",
    tone_col: str = "tone_score",
    prob_col: str = "sponsored_prob",
    bucket_col: str = "time_bucket",
) -> pd.DataFrame:
    """Generate signals for all (ticker, date) groups in the DataFrame (fully vectorized).

    Parameters
    ----------
    df : DataFrame
        Scored articles with ticker, effective_date, tone, sponsored_prob.

    Returns
    -------
    DataFrame with columns: ticker, effective_date, raw_signal, pred_score,
        direction, n_articles, n_organic, n_sponsored
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "ticker", "effective_date", "raw_signal", "pred_score",
            "direction", "n_articles", "n_organic", "n_sponsored",
        ])

    df_calc = df.copy()
    tones = df_calc[tone_col].fillna(0.0).to_numpy() if tone_col in df_calc else np.zeros(len(df_calc))
    probs = df_calc[prob_col].fillna(0.5).to_numpy() if prob_col in df_calc else np.full(len(df_calc), 0.5)
    buckets = df_calc[bucket_col].to_numpy() if bucket_col in df_calc else np.array(["OPEN"] * len(df_calc))
    is_closed = np.isin(buckets, ["CLOSED_POST", "CLOSED_PRE"])

    df_calc["_weighted_score"] = score_article(tones, probs, is_closed)
    df_calc["_is_organic"] = (probs < SPONSORED_PROB_LOW).astype(int)
    df_calc["_is_sponsored"] = (probs > SPONSORED_PROB_HIGH).astype(int)

    grouped = df_calc.groupby([ticker_col, date_col], as_index=False).agg(
        n_articles=("_weighted_score", "count"),
        raw_signal=("_weighted_score", "mean"),
        n_organic=("_is_organic", "sum"),
        n_sponsored=("_is_sponsored", "sum"),
    )

    # Filter/mask insufficient articles
    insufficient = grouped["n_articles"] < MIN_ARTICLES_FOR_SIGNAL
    grouped.loc[insufficient, "raw_signal"] = np.nan

    # Vectorized tanh pred_score
    grouped["pred_score"] = np.where(
        insufficient,
        np.nan,
        np.tanh(ALPHA * grouped["raw_signal"].fillna(0.0))
    )

    # Vectorized direction
    conditions = [
        insufficient,
        grouped["pred_score"] > DIRECTION_THRESHOLD,
        grouped["pred_score"] < -DIRECTION_THRESHOLD,
    ]
    choices = ["INSUFFICIENT_DATA", "BULLISH", "BEARISH"]
    grouped["direction"] = np.select(conditions, choices, default="NEUTRAL")

    col_order = [
        ticker_col, date_col,
        "raw_signal", "pred_score", "direction",
        "n_articles", "n_organic", "n_sponsored",
    ]
    result = grouped[[c for c in col_order if c in grouped.columns]]
    return result.sort_values([ticker_col, date_col]).reset_index(drop=True)

