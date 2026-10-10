"""Pipeline Step 14 — Periodic judge audit on the human-labeled subset.

Operationalizes the frozen-judge policy: re-score agreement between ANY
judge's verdicts and hand labels, append a timestamped audit record, and
enforce the kappa gate from pipeline/13_gpu_judge.py JUDGE_SPEC.

Usage:
  python pipeline/14_audit_judge.py                          # audit GPU judge
  python pipeline/14_audit_judge.py --verdicts data/final/local_judge_annotations.parquet --tag local-nli
  python pipeline/14_audit_judge.py --verdicts data/final/gpu_judge_annotations.parquet  --tag qwen3-32b

Output: outputs/tables/judge_audits.json (append-only audit trail)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from config.settings import DATA_FINAL, OUTPUTS_TABLES, setup_logging
from pipeline.human_validation import evaluate_agreement

logger = setup_logging()

AUDIT_PATH = OUTPUTS_TABLES / "judge_audits.json"


def audit(verdicts_path: Path, tag: str, min_kappa: float = 0.70) -> dict:
    lj = pd.read_parquet(verdicts_path)
    hands = pd.read_parquet(DATA_FINAL / "hand_labels.parquet")
    m = lj.merge(hands, left_on="url", right_on="source_url", how="inner")
    ev = pd.DataFrame({
        "hand_label": m["hand_label"].values.astype(int),
        "llm_label": m["numeric_label"].values,
        "weak_label": m["weak_label"].values if "weak_label" in m.columns else None,
    })
    ev = ev[ev["llm_label"].notna()].copy()
    ev["llm_label"] = ev["llm_label"].astype(int)
    ag = evaluate_agreement(ev, human_col="hand_label", llm_col="llm_label",
                            weak_col="weak_label")
    comp = ag["comparisons"].get("human_vs_llm", {})
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": tag,
        "verdicts": str(verdicts_path),
        "model": lj["model_name"].iloc[0] if "model_name" in lj.columns and len(lj) else tag,
        "n": comp.get("n", len(ev)),
        "kappa": comp.get("cohen_kappa"),
        "f1": comp.get("f1_score"),
        "min_kappa_gate": min_kappa,
        "pass": bool(comp.get("cohen_kappa", -1) >= min_kappa),
    }
    trail = json.loads(AUDIT_PATH.read_text()) if AUDIT_PATH.exists() else []
    trail.append(record)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(trail, indent=2))
    logger.info(f"Audit [{tag}]: kappa={record['kappa']:.3f} f1={record['f1']:.3f} "
                f"-> {'PASS' if record['pass'] else 'FAIL (do not ship)'}")
    return record


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit a judge against hand labels.")
    p.add_argument("--verdicts", type=str,
                   default=str(DATA_FINAL / "gpu_judge_annotations.parquet"))
    p.add_argument("--tag", type=str, default="qwen3-32b")
    p.add_argument("--min-kappa", type=float, default=0.70)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    rec = audit(Path(a.verdicts), a.tag, a.min_kappa)
    if not rec["pass"]:
        raise SystemExit(f"AUDIT FAIL: kappa {rec['kappa']:.3f} < gate {a.min_kappa}")
