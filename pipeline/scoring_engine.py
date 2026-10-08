"""Sponsored News Scoring Engine (HHEE).

Implements the Hierarchical Hybrid Evidence Engine:
  - Layer 0: Deterministic Invariant Gates (PR Wires, Explicit Sponsored URLs, Disclosures)
  - Layer 1: Continuous Observable Evidence Aggregation S(A) in [0, 1]
  - Layer 2: Precision-Constrained Calibrated Thresholding with Abstain Zone
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from pipeline.extractors.news_morphometrics import (
    extract_all_morphometrics,
    extract_boilerplate_ratio,
    extract_cta_density,
    extract_disclosure_flag,
    extract_headline_hype,
    extract_pr_wire_flag,
    extract_promo_density,
    extract_quote_monopoly,
    extract_risk_absence,
    extract_url_sponsored_flag,
)

# Layer 1 weights — LEARNED v2 (2026-10-09, n=149 real scraped bodies, text-only, no URL/domain).
# Method: L1 LogisticRegressionCV(Cs=[0.1,0.5,1,2,5], cv=5) on hand labels, C=2 selected.
# Stable 5-fold CV AUC = 0.688 (matches metadata-only honest AUC 0.705; no lift without domains).
# NOTE: unregularized fit (C=10000) reached CV AUC 0.778 but with unstable cta_density coef
# (+132.6, rare feature, 0.000 vs 0.005 means) — rejected as overfit, see outputs/tables/hhee_v2_weights.json.
# Previous hand-set v1 kept below for provenance.
HAND_SET_V1_WEIGHTS: Dict[str, float] = {
    "promo_density": 2.5,
    "cta_density": 3.2,
    "boilerplate_ratio": 2.2,
    "headline_hype": 1.4,
    "risk_absence": 1.1,
    "quote_monopoly": 0.8,
}
HAND_SET_V1_BIAS = -2.8
DEFAULT_WEIGHTS: Dict[str, float] = {
    "promo_density": 0.99,
    "cta_density": 0.234,
    "boilerplate_ratio": 2.17,
    "headline_hype": 0.881,
    "risk_absence": -0.366,
    "quote_monopoly": -1.255,
}
DEFAULT_BIAS = -0.692


def sigmoid(z: float) -> float:
    """Standard numerically stable logistic sigmoid."""
    if z < -500:
        return 0.0
    if z > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def score_single_article(
    url: str = "",
    headline: str = "",
    text: str = "",
    domain: str = "",
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
    bias: float = DEFAULT_BIAS,
    tau_low: float = 0.30,
    tau_high: float = 0.70,
) -> Dict[str, Any]:
    """Score an individual article through the 3-layer HHEE architecture.
    
    Returns
    -------
    dict containing:
      - score: float in [0, 1]
      - classification: 'sponsored' | 'organic' | 'unclear'
      - layer_triggered: 'Layer 0 (Invariant)' | 'Layer 1 (Continuous Evidence)'
      - attribution: dict mapping feature name to additive logit contribution
      - features: dict of raw extracted morphometrics
    """
    feats = extract_all_morphometrics(url=url, headline=headline, text=text, domain=domain)

    # ── Layer 0: Deterministic Invariants ─────────────────────────────
    if feats["url_sponsor_flag"] >= 1.0:
        return {
            "score": 1.0,
            "classification": "sponsored",
            "layer_triggered": "Layer 0 (URL Sponsored Path)",
            "attribution": {"url_sponsor_flag": 10.0},
            "features": feats,
        }
    if feats["pr_wire_flag"] >= 1.0:
        return {
            "score": 1.0,
            "classification": "sponsored",
            "layer_triggered": "Layer 0 (PR Wire Origin)",
            "attribution": {"pr_wire_flag": 10.0},
            "features": feats,
        }
    if feats.get("market_research_spam_flag", 0.0) >= 1.0:
        return {
            "score": 1.0,
            "classification": "sponsored",
            "layer_triggered": "Layer 0 (Market Research PR Spam)",
            "attribution": {"market_research_spam_flag": 10.0},
            "features": feats,
        }
    if feats["disclosure_flag"] >= 1.0:
        return {
            "score": 1.0,
            "classification": "sponsored",
            "layer_triggered": "Layer 0 (Statutory Disclosure)",
            "attribution": {"disclosure_flag": 10.0},
            "features": feats,
        }

    # ── Layer 1: Continuous Evidence Aggregation ──────────────────────
    logit = bias
    attribution: Dict[str, float] = {"bias": bias}

    for feat_name, w in weights.items():
        val = feats.get(feat_name, 0.0)
        contrib = w * val
        logit += contrib
        attribution[feat_name] = round(contrib, 4)

    score = sigmoid(logit)

    # ── Layer 2: Calibrated Thresholding with Abstain Zone ────────────
    if score >= tau_high:
        cls_label = "sponsored"
    elif score <= tau_low:
        cls_label = "organic"
    else:
        cls_label = "unclear"

    return {
        "score": round(score, 6),
        "classification": cls_label,
        "layer_triggered": "Layer 1 (Evidence Score)",
        "attribution": attribution,
        "features": feats,
    }


def score_dataframe(
    df: pd.DataFrame,
    tau_low: float = 0.30,
    tau_high: float = 0.70,
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
    bias: float = DEFAULT_BIAS,
) -> pd.DataFrame:
    """Vectorized / batch scoring of a DataFrame of news articles."""
    out = df.copy()

    scores = []
    classes = []
    layers = []
    promo_densities = []
    cta_densities = []
    boilerplates = []
    risk_absences = []
    top_signals = []

    for idx, row in out.iterrows():
        url_val = str(row.get("source_url", "") or "")
        domain_val = str(row.get("source_domain", "") or "")
        title_val = str(row.get("title", "") or "")
        text_val = str(row.get("content", "") or row.get("article_text", "") or url_val)

        res = score_single_article(
            url=url_val,
            headline=title_val,
            text=text_val,
            domain=domain_val,
            weights=weights,
            bias=bias,
            tau_low=tau_low,
            tau_high=tau_high,
        )

        scores.append(res["score"])
        classes.append(res["classification"])
        layers.append(res["layer_triggered"])
        promo_densities.append(res["features"]["promo_density"])
        cta_densities.append(res["features"]["cta_density"])
        boilerplates.append(res["features"]["boilerplate_ratio"])
        risk_absences.append(res["features"]["risk_absence"])

        # Determine top 2 positive drivers
        attribs = sorted(
            [(k, v) for k, v in res["attribution"].items() if k != "bias"],
            key=lambda x: x[1],
            reverse=True,
        )
        top_str = "; ".join(f"{k}:{v:+.2f}" for k, v in attribs[:2] if v > 0)
        top_signals.append(top_str or "none")

    out["sponsored_prob"] = scores
    out["sponsored_score"] = scores
    out["sponsored_class"] = classes
    out["layer_triggered"] = layers
    out["promo_density"] = promo_densities
    out["cta_density"] = cta_densities
    out["boilerplate_ratio"] = boilerplates
    out["risk_absence"] = risk_absences
    out["top_signals"] = top_signals

    return out
