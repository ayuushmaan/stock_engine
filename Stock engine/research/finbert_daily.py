"""Daily transformer-sentiment signals from URL-slug pseudo-titles.

Production counterpart of research/05_finbert_validation.py (which is a
jupytext notebook). Scores every row of sponsored_scores.parquet with a
finance transformer, aggregates to (effective_date, ticker) daily means,
and writes data/processed/finbert_daily.parquet for the baselines runner.

Honesty notes (do not remove):
  - GDELT GKG stores NO headlines; pseudo-titles come from URL slugs.
    Parse success rate is reported; unparseable rows are dropped, not imputed.
  - ProsusAI/finbert ships only pytorch_model.bin, which transformers
    refuses to load on torch<2.6 (CVE-2025-32434 gate). We try it with
    use_safetensors=True and fall back to the safetensors-native
    mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis,
    which is recorded in the output JSON. Both are finance-sentiment
    BERT-family models; the fallback is a *baseline*, not "FinBERT".
  - Coverage is bounded by sponsored_scores.parquet (currently Jan-Feb
    2020, ~20k rows). Thin daily cross-sections are reported, not hidden.

Usage:
  python research/finbert_daily.py
  python research/finbert_daily.py --limit 500   # smoke test
Output:
  data/processed/finbert_daily.parquet        (effective_date, ticker, finbert_sentiment, n_articles)
  data/processed/finbert_article_scores.parquet (article-level audit trail)
  outputs/tables/finbert_daily_meta.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer

from config.settings import DATA_PROCESSED, OUTPUTS_TABLES, setup_logging

logger = setup_logging()

CANDIDATE_MODELS = [
    ("ProsusAI/finbert", True),  # preferred name; safetensors often absent -> fallback
    ("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", False),
]


def extract_title_from_url(url: str) -> str | None:
    """Pseudo-title from URL slug (same rules as 05_finbert_validation)."""
    if not isinstance(url, str) or len(url) < 10:
        return None
    try:
        parsed = urlparse(url)
        path = re.sub(r'\.(html?|cms|php|asp|aspx|jsp|xml|json)$', '',
                      parsed.path.rstrip('/'), flags=re.IGNORECASE)
        slug = path.split('/')[-1]
        if slug.isdigit() or slug.lower() == 'articleshow':
            for part in reversed(path.split('/')):
                if not part.isdigit() and part.lower() != 'articleshow' and len(part) > 3:
                    slug = part
                    break
        slug = re.sub(r'[-_]?\d{5,}$', '', slug)
        slug = re.sub(r'^(article|story|news|post|blog)[-_]', '', slug, flags=re.IGNORECASE)
        title = re.sub(r'[-_]+', ' ', slug).strip()
        if len(title) < 8 or title.replace(' ', '').isdigit():
            return None
        return title
    except Exception:
        return None


def load_transformer():
    """First working finance-sentiment model; returns (name, tok, model, device)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for name, wants_safe in CANDIDATE_MODELS:
        try:
            try:
                tok = AutoTokenizer.from_pretrained(name)
            except ValueError:
                tok = BertTokenizer.from_pretrained(name)  # slow-tokenizer fallback
            kw = {"use_safetensors": True} if wants_safe else {}
            mdl = AutoModelForSequenceClassification.from_pretrained(name, **kw)
            mdl.eval().to(device)
            logger.info(f"Transformer scorer: {name} on {device}")
            return name, tok, mdl, device
        except Exception as e:
            logger.warning(f"{name} unusable ({str(e)[:120]}); trying next.")
    raise RuntimeError("No transformer sentiment model loadable in this env.")


def score_texts(texts: list[str], tok, mdl, device, batch: int = 64) -> np.ndarray:
    """P(positive) - P(negative); label-agnostic via id2label mapping."""
    id2label = {int(k): str(v).lower() for k, v in mdl.config.id2label.items()}
    pos_idx = next(i for i, l in id2label.items() if 'pos' in l)
    neg_idx = next(i for i, l in id2label.items() if 'neg' in l)
    import torch.nn.functional as F
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=128, return_tensors='pt').to(device)
            probs = F.softmax(mdl(**enc).logits, dim=-1).cpu().numpy()
            outs.append(probs[:, pos_idx] - probs[:, neg_idx])
    return np.concatenate(outs)


def run(limit: int | None = None, full: bool = True) -> pd.DataFrame:
    # Full 1.4M-row labeled panel by default (same columns); 20k scored
    # subset only as fallback or with --no-full. Coverage, not sampling.
    src = DATA_PROCESSED / "gdelt_labeled.parquet"
    if full is False or not src.exists():
        src = DATA_PROCESSED / "sponsored_scores.parquet"
    if not src.exists():
        raise FileNotFoundError("No labeled panel — run pipeline steps 01-04 first.")
    logger.info(f"Source panel: {src.name}")
    df = pd.read_parquet(src)
    tone_col = "tone" if "tone" in df.columns else "tone_score"
    url_col = "DocumentIdentifier" if "DocumentIdentifier" in df.columns else "source_url"
    df = df.rename(columns={tone_col: "tone_score"})
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if limit:
        df = df.head(limit)
    logger.info(f"Input articles: {len(df):,}")

    df["pseudo_title"] = df[url_col].apply(extract_title_from_url)
    n_parse = int(df["pseudo_title"].notna().sum())
    logger.info(f"Slug parse success: {n_parse:,}/{len(df):,} ({100*n_parse/len(df):.1f}%)")
    df = df[df["pseudo_title"].notna()].copy()
    if df.empty:
        raise RuntimeError("Zero parseable pseudo-titles.")

    name, tok, mdl, device = load_transformer()
    df["finbert_sentiment"] = score_texts(df["pseudo_title"].tolist(), tok, mdl, device)
    logger.info(f"finbert_sentiment describe:\n{df['finbert_sentiment'].describe().to_string()}")

    art_path = DATA_PROCESSED / "finbert_article_scores.parquet"
    df[["ticker", "effective_date", "time_bucket", "pseudo_title",
        "tone_score", "finbert_sentiment"]].to_parquet(art_path, index=False)

    df["effective_date"] = pd.to_datetime(df["effective_date"])
    daily = (df.groupby(["effective_date", "ticker"])
               .agg(finbert_sentiment=("finbert_sentiment", "mean"),
                    n_articles=("finbert_sentiment", "size"))
               .reset_index())
    out_path = DATA_PROCESSED / "finbert_daily.parquet"
    daily.to_parquet(out_path, index=False)

    meta = {
        "model": name, "device": str(device),
        "n_input": len(df), "parse_rate": round(n_parse / len(df), 4) if len(df) else 0.0,
        "n_daily_cells": len(daily),
        "median_articles_per_cell": float(daily["n_articles"].median()) if len(daily) else 0.0,
        "date_min": str(daily["effective_date"].min()) if len(daily) else None,
        "date_max": str(daily["effective_date"].max()) if len(daily) else None,
        "caveats": ("URL-slug pseudo-titles, not headlines; coverage bounded by "
                    "sponsored_scores.parquet; ProsusAI/finbert .bin blocked on torch<2.6."),
    }
    with open(OUTPUTS_TABLES / "finbert_daily_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Daily cells: {len(daily):,} -> {out_path}")
    return daily


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily transformer sentiment signals.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-full", action="store_true", help="Use 20k scored subset only")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(limit=a.limit, full=not a.no_full)
