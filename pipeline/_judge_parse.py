"""Shared output parser for local LLM judges (steps 12/13)."""
from __future__ import annotations

import json
import re


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
