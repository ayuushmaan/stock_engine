"""Pipeline Step 13 — GPU LLM-as-Judge (4-bit, RTX 3050 6GB).

Qwen/Qwen2.5-7B-Instruct loaded NF4 4-bit (~4.5GB VRAM) with greedy
decode. Same LLMAnnotation schema as pipeline/llm_labeler.py.

Why 7B-Instruct and not DeepSeek-R1-Distill: reasoning models emit long
<think> traces (slow on 6GB) built for math/code, not 3-way media
classification; an instruct model with JSON mode is the sharper judge
here. R1-Distill-Qwen-7B can be swapped via --model if desired.

Inputs:  data/final/article_bodies.parquet (word_count_scraped > 50)
Outputs: data/final/gpu_judge_annotations.parquet
Cache:   data/processed/gpu_judge_cache/ (per-model SHA-256, resumable)

Usage:
  python pipeline/13_gpu_judge.py --limit 3
  python pipeline/13_gpu_judge.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config.settings import DATA_FINAL, DATA_PROCESSED, setup_logging
from pipeline.llm_labeler import PROMPT_SYSTEM
from pipeline._judge_parse import parse_label

logger = setup_logging()

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# On 96GB cards (MoLab RTX PRO 6000) the strict pick is Qwen3-32B in fp16:
# verified Oct 2026: kappa 0.75 vs 200 hand labels (F1 0.81). Ungated.
MOLAB_MODEL = "Qwen/Qwen3-32B"
CACHE_DIR = DATA_PROCESSED / "gpu_judge_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MAX_BODY_CHARS = 900


def _cache_path(model: str, url: str) -> Path:
    key = hashlib.sha256(f"{model}:{url}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{key}.json"


def run(model: str = DEFAULT_MODEL, limit: int | None = None, quant_4bit: bool = True) -> pd.DataFrame:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU — use pipeline/12_local_judge.py (CPU) instead.")
    bodies = pd.read_parquet(DATA_FINAL / "article_bodies.parquet")
    df = bodies[bodies["word_count_scraped"] > 50].copy()
    if limit:
        df = df.head(limit)
    mode = "NF4 4-bit" if quant_4bit else "fp16 full"
    logger.info(f"GPU-judge scoring {len(df)} articles with {model} ({mode})")

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    if quant_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
        mdl = AutoModelForCausalLM.from_pretrained(
            model, quantization_config=bnb, device_map="auto",
            torch_dtype=torch.float16, trust_remote_code=False,
        )
    else:
        mdl = AutoModelForCausalLM.from_pretrained(
            model, device_map="auto", torch_dtype=torch.float16,
            trust_remote_code=False,
        )
    mdl.eval()
    logger.info(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f}GB / "
                f"{torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")

    rows, t0 = [], time.time()
    for i, r in enumerate(df.itertuples()):
        cp = _cache_path(model, r.url)
        if cp.exists():
            try:
                rows.append(json.loads(cp.read_text(encoding="utf-8")))
                continue
            except Exception:
                pass
        domain = r.url.split("/")[2] if "://" in r.url else ""
        user = (f"Headline: {r.title}\nDomain: {domain}\nURL: {r.url}\n"
                f"Body: {(r.body or '')[:MAX_BODY_CHARS]}")
        messages = [{"role": "system", "content": PROMPT_SYSTEM},
                    {"role": "user", "content": user}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=200, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = parse_label(gen)
        rec = {
            "article_id": hashlib.md5(r.url.encode()).hexdigest()[:12],
            "url": r.url, "domain": domain, "title": r.title,
            "label": parsed["label"],
            "numeric_label": 1 if parsed["label"] == "SPONSORED" else (0 if parsed["label"] == "ORGANIC" else None),
            "confidence": parsed["confidence"], "reasoning": parsed["reasoning"],
            "detected_signals": parsed["detected_signals"],
            "model_name": "gpu4bit:" + model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        cp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        rows.append(rec)
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            logger.info(f"  {i+1}/{len(df)} ({el/(i+1):.1f}s/article, "
                        f"eta {(len(df)-i-1)*el/(i+1)/60:.0f}m)")

    out = pd.DataFrame(rows)
    out_path = DATA_FINAL / "gpu_judge_annotations.parquet"
    out.to_parquet(out_path, index=False)
    logger.info(f"Saved {out_path} ({len(out)} rows)")
    logger.info(f"Distribution:\n{out['label'].value_counts(dropna=False)}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU LLM judge (4-bit or fp16).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--fp16", action="store_true",
                   help="Disable 4-bit quant (needs 16GB+ VRAM for 7B, 64GB+ for 32B)")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(model=a.model, limit=a.limit, quant_4bit=not a.fp16)
