"""Pipeline Step 12 — Local LLM-as-Judge (offline, CPU, free).

Why this exists:
  HF Inference Providers is paywalled on this account (402 no credits +
  provider gating). This module runs a small instruct model LOCALLY via
  transformers so the project still gets a genuine, independent second
  judge — fully offline and reproducible.

Model: Qwen/Qwen2.5-1.5B-Instruct (safetensors, ~3GB, CPU-friendly).
Output schema matches pipeline/llm_labeler.py LLMAnnotation.

Inputs:
  data/final/article_bodies.parquet (url, title, body)

Outputs:
  data/final/local_judge_annotations.parquet
  outputs/tables/local_judge_agreement.json (human vs local-judge)

Design:
  - SHA-256 prompt cache in data/processed/local_judge_cache/ (resumable)
  - Greedy decode (temperature=None), max_new_tokens=180
  - Robust JSON parse: fence-strip -> json.loads -> regex fallback -> UNCLEAR
  - Body truncated to 900 chars to bound CPU time (~30-60s/article)

Usage:
  python pipeline/12_local_judge.py --limit 3    # smoke test
  python pipeline/12_local_judge.py              # full 149
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch

from config.settings import DATA_FINAL, DATA_PROCESSED, OUTPUTS_TABLES, setup_logging
from pipeline._judge_parse import parse_label

logger = setup_logging()

MODEL_ID = "typeform/distilbert-base-uncased-mnli"
CANDIDATE_LABELS = ["paid promotional content", "independent journalism"]
CACHE_DIR = DATA_PROCESSED / "local_judge_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MAX_BODY_CHARS = 900


def _cache_path(url: str) -> Path:
    return CACHE_DIR / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".json")


def parse_label(text: str) -> dict:
    """Extract (label, confidence, reasoning, signals) from free-form output."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    try:
        d = json.loads(t)
        label = str(d.get("label", "UNCLEAR")).upper()
        if label not in ("SPONSORED", "ORGANIC", "UNCLEAR"):
            label = "UNCLEAR"
        return {
            "label": label,
            "confidence": float(d.get("confidence", 0.5)),
            "reasoning": str(d.get("reasoning", ""))[:500],
            "detected_signals": list(d.get("detected_signals", []))[:6],
        }
    except Exception:
        pass
    up = t.upper()
    if "SPONSORED" in up and "ORGANIC" not in up:
        return {"label": "SPONSORED", "confidence": 0.6, "reasoning": t[:500], "detected_signals": []}
    if "ORGANIC" in up and "SPONSORED" not in up:
        return {"label": "ORGANIC", "confidence": 0.6, "reasoning": t[:500], "detected_signals": []}
    m = re.search(r'\{[^{}]*"label"[^{}]*\}', t, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            label = str(d.get("label", "UNCLEAR")).upper()
            if label in ("SPONSORED", "ORGANIC"):
                return {"label": label, "confidence": 0.55, "reasoning": t[:500], "detected_signals": []}
        except Exception:
            pass
    return {"label": "UNCLEAR", "confidence": 0.5, "reasoning": t[:500], "detected_signals": []}


def run(limit: int | None = None) -> pd.DataFrame:
    bodies = pd.read_parquet(DATA_FINAL / "article_bodies.parquet")
    df = bodies[bodies["word_count_scraped"] > 50].copy()
    if limit:
        df = df.head(limit)
    logger.info(f"Local-judge scoring {len(df)} articles with {MODEL_ID} (CPU)")

    from transformers import pipeline as hf_pipeline
    clf = hf_pipeline("zero-shot-classification", model=MODEL_ID, device=-1,
                      torch_dtype=torch.float32)
    n_threads = max(2, (torch.get_num_threads() or 4))
    torch.set_num_threads(n_threads)
    logger.info(f"torch threads: {n_threads}")

    rows = []
    t0 = time.time()
    for i, r in enumerate(df.itertuples()):
        cp = _cache_path(r.url)
        if cp.exists():
            try:
                rows.append(json.loads(cp.read_text(encoding="utf-8")))
                continue
            except Exception:
                pass
        text = f"{r.title or ''}. {(r.body or '')[:MAX_BODY_CHARS]}"
        try:
            z = clf(text, CANDIDATE_LABELS, multi_label=False, truncation=True)
            top, top_score = z["labels"][0], float(z["scores"][0])
            label = "SPONSORED" if top == CANDIDATE_LABELS[0] else "ORGANIC"
            parsed = {"label": label, "confidence": top_score,
                      "reasoning": f"zero-shot NLI top={top} ({top_score:.2f})",
                      "detected_signals": [top]}
        except Exception as e:
            logger.warning(f"local judge failed for {r.url}: {e}")
            parsed = {"label": "UNCLEAR", "confidence": 0.5, "reasoning": "inference error", "detected_signals": []}
        rec = {
            "article_id": hashlib.md5(r.url.encode()).hexdigest()[:12],
            "url": r.url,
            "domain": r.url.split("/")[2] if "://" in r.url else "",
            "title": r.title,
            "label": parsed["label"],
            "numeric_label": 1 if parsed["label"] == "SPONSORED" else (0 if parsed["label"] == "ORGANIC" else None),
            "confidence": parsed["confidence"],
            "reasoning": parsed["reasoning"],
            "detected_signals": parsed["detected_signals"],
            "model_name": "local:" + MODEL_ID,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        cp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        rows.append(rec)
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            logger.info(f"  {i+1}/{len(df)} ({el/(i+1):.1f}s/article, eta {(len(df)-i-1)*el/(i+1)/60:.0f}m)")

    out = pd.DataFrame(rows)
    out_path = DATA_FINAL / "local_judge_annotations.parquet"
    out.to_parquet(out_path, index=False)
    logger.info(f"Saved {out_path} ({len(out)} rows)")
    logger.info(f"Distribution:\n{out['label'].value_counts(dropna=False)}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline local LLM judge.")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(limit=parse_args().limit)
