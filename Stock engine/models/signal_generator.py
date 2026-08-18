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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    tone: float,
    sponsored_prob: float,
    is_closed_window: bool,
) -> float:
    """Compute weighted score for a single article.

    Parameters
    ----------
    tone : float
        Raw GDELT tone score (typically [-20, +20]).
    sponsored_prob : float
        Probability that this article is sponsored (0-1).
    is_closed_window : bool
        True if the article was published during market-closed hours.

    Returns
    -------
    float
        Weighted article score.
    """
    article_score = tone / TONE_NORMALIZER
    organic_weight = max(1.0 - sponsored_prob, ORGANIC_WEIGHT_FLOOR)
    time_weight = TIME_WEIGHT_CLOSED if is_closed_window else TIME_WEIGHT_OPEN
    return article_score * organic_weight * time_weight


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
    if len(articles) < MIN_ARTICLES_FOR_SIGNAL:
        return {
            "raw_signal": np.nan,
            "pred_score": np.nan,
            "direction": "INSUFFICIENT_DATA",
            "n_articles": len(articles),
            "n_organic": 0,
            "n_sponsored": 0,
        }

    weighted_scores = []
    for _, row in articles.iterrows():
        tone = row.get(tone_col, 0) or 0
        prob = row.get(prob_col, 0.5) or 0.5
        bucket = row.get(bucket_col, "OPEN")
        is_closed = bucket in ("CLOSED_POST", "CLOSED_PRE")
        weighted_scores.append(score_article(tone, prob, is_closed))

    raw_signal = float(np.mean(weighted_scores))
    pred_score = float(np.tanh(ALPHA * raw_signal))

    if pred_score > DIRECTION_THRESHOLD:
        direction = "BULLISH"
    elif pred_score < -DIRECTION_THRESHOLD:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    return {
        "raw_signal": raw_signal,
        "pred_score": pred_score,
        "direction": direction,
        "n_articles": len(articles),
        "n_organic": int((articles[prob_col] < SPONSORED_PROB_LOW).sum()),
        "n_sponsored": int((articles[prob_col] > SPONSORED_PROB_HIGH).sum()),
    }


def batch_generate_signals(
    df: pd.DataFrame,
    ticker_col: str = "ticker",
    date_col: str = "effective_date",
    tone_col: str = "tone_score",
    prob_col: str = "sponsored_prob",
    bucket_col: str = "time_bucket",
) -> pd.DataFrame:
    """Generate signals for all (ticker, date) groups in the DataFrame.

    Parameters
    ----------
    df : DataFrame
        Scored articles with ticker, effective_date, tone, sponsored_prob.

    Returns
    -------
    DataFrame with columns: ticker, effective_date, raw_signal, pred_score,
        direction, n_articles, n_organic, n_sponsored
    """
    records = []
    for (ticker, date), group in df.groupby([ticker_col, date_col]):
        signal = generate_signal(group, tone_col, prob_col, bucket_col)
        signal["ticker"] = ticker
        signal["effective_date"] = date
        records.append(signal)

    result = pd.DataFrame(records)
    col_order = [
        "ticker", "effective_date",
        "raw_signal", "pred_score", "direction",
        "n_articles", "n_organic", "n_sponsored",
    ]
    result = result[[c for c in col_order if c in result.columns]]
    return result.sort_values(["ticker", "effective_date"]).reset_index(drop=True)
