# ML-Trained Configuration for Sponsored News Penalty
# Generated: 2026-04-21 16:08:08 IST

# Model parameters learned from 10-year historical backtest
SPONSORED_NEWS_PENALTY = 0.449
CALIBRATION_K = 3.327
IMPROVEMENT_OVER_BASELINE = 3.3%

# Interpretation:
# - Apply a penalty of 44.9% to sponsored news sentiment
#   (multiply by 1 - 0.449 = 0.551)
# - Use calibration k=3.327 to convert sentiment to % move
# - This improves RMSE by 3.3% vs baseline

# Formula:
# adjusted_sentiment = sentiment * (1 - 0.449 * is_sponsored)
# expected_pct_change = 3.327 * tanh(1.5 * adjusted_sentiment)
