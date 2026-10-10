"""Pipeline Step 15 — Daily paper-trading loop (NO real money, NO order execution).

Design (the honest live test):
  predict: score the latest news date -> emit next-day LONG/SHORT paper
           predictions, appended to data/final/paper_predictions.parquet
  score:   join logged predictions with realized forwards -> rolling
           hit-rate / IC scoreboard in outputs/tables/paper_scoreboard.json

  Predictions are POINT-IN-TIME (logged before the predicted session) so
  this loop cannot be gamed ex-post. It replaces *claims* about/call for
  live capital with a verifiable prediction diary.

  Current verdict (Oct 2026): signal IC ~0.005 ns, net -22%/yr after costs.
  DO NOT attach real capital. This loop exists to EARN the right to.

Usage:
  python pipeline/15_paper_trade.py predict [--date YYYY-MM-DD] [--top-n 5]
  python pipeline/15_paper_trade.py score
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import DATA_FINAL, DATA_PROCESSED, OUTPUTS_TABLES, setup_logging
from models.signal_generator import batch_generate_signals

logger = setup_logging()
PRED_PATH = DATA_FINAL / "paper_predictions.parquet"
SCORE_PATH = OUTPUTS_TABLES / "paper_scoreboard.json"


def predict(asof: str | None = None, top_n: int = 5) -> pd.DataFrame:
    scored = DATA_PROCESSED / "sponsored_scores.parquet"
    if not scored.exists():
        raise FileNotFoundError(f"{scored} missing — run pipeline steps 01-04 first.")
    df = pd.read_parquet(scored)
    date_col = "effective_date" if "effective_date" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col])
    target = pd.to_datetime(asof) if asof else df[date_col].max()
    day = df[df[date_col] == target].copy()
    if day.empty:
        raise ValueError(f"No scored news for {target.date()}.")
    if "sponsored_prob" not in day.columns and "sponsored_score" in day.columns:
        day["sponsored_prob"] = day["sponsored_score"]
    if "tone_score" not in day.columns and "tone" in day.columns:
        day["tone_score"] = day["tone"]

    sigs = batch_generate_signals(day)
    sigs = sigs[sigs["direction"].isin(["BULLISH", "BEARISH"])].copy()
    longs = sigs.nlargest(top_n, "pred_score")
    shorts = sigs.nsmallest(top_n, "pred_score")
    recs = []
    for _, r in longs.iterrows():
        recs.append({"predict_date": target, "ticker": r["ticker"],
                     "side": "LONG", "pred_score": float(r["pred_score"]),
                     "n_articles": int(r["n_articles"])})
    for _, r in shorts.iterrows():
        recs.append({"predict_date": target, "ticker": r["ticker"],
                     "side": "SHORT", "pred_score": float(r["pred_score"]),
                     "n_articles": int(r["n_articles"])})
    out = pd.DataFrame(recs)
    out["logged_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if PRED_PATH.exists():
        old = pd.read_parquet(PRED_PATH)
        out = pd.concat([old, out], ignore_index=True).drop_duplicates(
            subset=["predict_date", "ticker", "side"])
    out.to_parquet(PRED_PATH, index=False)
    logger.info(f"Logged {len(recs)} paper predictions for next session after {target.date()}")
    return out


def score() -> dict:
    if not PRED_PATH.exists():
        raise FileNotFoundError("No predictions logged yet — run `predict` first.")
    preds = pd.read_parquet(PRED_PATH)
    preds["predict_date"] = pd.to_datetime(preds["predict_date"])
    master = pd.read_parquet(DATA_FINAL / "master_dataset.parquet")
    dcol = "date" if "date" in master.columns else "effective_date"
    master[dcol] = pd.to_datetime(master[dcol])
    master = master.sort_values(["ticker", dcol])
    master["fwd1"] = master.groupby("ticker")["ret_close2close"].shift(-1)
    m = preds.merge(master[["ticker", dcol, "fwd1"]].rename(columns={dcol: "predict_date"}),
                    on=["ticker", "predict_date"], how="left")
    scored = m.dropna(subset=["fwd1"]).copy()
    if scored.empty:
        return {"n_scored": 0, "note": "predictions too recent — forwards not realized yet"}
    scored["pnl_side"] = np.where(scored["side"] == "LONG", scored["fwd1"], -scored["fwd1"])
    by_day = scored.groupby("predict_date")["pnl_side"].mean()
    res = {
        "n_predictions": len(preds), "n_scored": len(scored),
        "n_days": len(by_day),
        "hit_rate": round(float((scored["pnl_side"] > 0).mean()), 4),
        "mean_daily_paper_return_pct": round(float(by_day.mean() * 100), 4),
        "ann_paper_return_pct": round(float(by_day.mean() * 252 * 100), 2) if len(by_day) >= 20 else None,
        "note": ("PAPER ONLY — no capital, no execution. Gross of costs. "
                 "Annualized figure suppressed until >=20 scored days."),
    }
    SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCORE_PATH.write_text(json.dumps(res, indent=2))
    logger.info(f"Scoreboard: {res}")
    return res


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper-trading prediction diary.")
    p.add_argument("mode", choices=["predict", "score"])
    p.add_argument("--date", default=None)
    p.add_argument("--top-n", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.mode == "predict":
        predict(a.date, a.top_n)
    else:
        print(json.dumps(score(), indent=2))
