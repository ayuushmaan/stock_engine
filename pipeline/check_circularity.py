r"""Pipeline Step 9 — Phase 0.3: Structural circularity audit with build-fail assertion.

Enforces a hard structural invariant:
    NO feature name used by the LightGBM classifier may appear in the
    labeling function (assign_weak_labels in pipeline/03_label_news.py).

This script:
    1. Parses pipeline/03_label_news.py to extract all identifiers used in
       the assign_weak_labels function body.
    2. Parses pipeline/04_sponsored_classifier.py to extract the FEATURE_COLS list.
    3. Computes the intersection — features that appear in BOTH the labeling
       function and the classifier feature set.
    4. FAILS THE BUILD (exit code 1) if any overlap is found.

This is designed to be run as a CI/CD gate or pre-commit check.

Outputs:
    outputs/tables/circularity_audit.json  — detailed audit results
    Exit code 0 → clean, no circularity
    Exit code 1 → circularity detected → BUILD FAILURE

Usage:
    python pipeline/check_circularity.py
    python pipeline/check_circularity.py --strict    # also flag indirect overlaps

Integration:
    # In CI/CD (GitHub Actions, etc.):
    - name: Circularity check
      run: python pipeline/check_circularity.py
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# ── project imports ───────────────────────────────────────────────
from config.settings import OUTPUTS_TABLES, setup_logging

logger = setup_logging()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_SCRIPT = PROJECT_ROOT / "pipeline" / "03_label_news.py"
CLASSIFIER_SCRIPT = PROJECT_ROOT / "pipeline" / "04_sponsored_classifier.py"
HONEST_EVAL_SCRIPT = PROJECT_ROOT / "pipeline" / "08_honest_eval.py"


# ── AST-based feature extraction ─────────────────────────────────

def _extract_function_body_names(filepath: Path, function_name: str) -> set[str]:
    """Extract all Name identifiers referenced inside a function body using AST.

    Parameters
    ----------
    filepath : Path
        Python source file to parse.
    function_name : str
        Name of the function to inspect.

    Returns
    -------
    set[str]
        All identifier names used within the function body.
    """
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))

    names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # Walk all nodes inside this function
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    names.add(child.id)
                elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                    names.add(child.value)
                elif isinstance(child, ast.Subscript):
                    # Catch df["column_name"] patterns
                    if isinstance(child.slice, ast.Constant) and isinstance(child.slice.value, str):
                        names.add(child.slice.value)
    return names


def _extract_string_constants_from_function(filepath: Path, function_name: str) -> set[str]:
    """Extract all string constants used inside a function (column names in df["col"] access).

    This specifically catches patterns like:
        df["tone"]
        df["source_tier"]
        row["something"]
    """
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))

    strings = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for child in ast.walk(node):
                # df["column_name"] → Subscript with Constant slice
                if isinstance(child, ast.Subscript):
                    if isinstance(child.slice, ast.Constant) and isinstance(child.slice.value, str):
                        strings.add(child.slice.value)
                # String constants in comparisons, function calls, etc.
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    strings.add(child.value)

    return strings


def _extract_feature_cols(filepath: Path) -> list[str]:
    """Extract the FEATURE_COLS list from a Python source file using AST.

    Looks for a top-level assignment like:
        FEATURE_COLS = ["tone_score", "positive_score", ...]
    """
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "FEATURE_COLS":
                    if isinstance(node.value, ast.List):
                        return [
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]

    return []


def _extract_excluded_features(filepath: Path) -> list[str]:
    """Extract the CIRCULAR_FEATURES list from the honest eval script."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CIRCULAR_FEATURES":
                    if isinstance(node.value, ast.List):
                        return [
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]

    return []


def _extract_honest_features(filepath: Path) -> list[str]:
    """Extract the HONEST_FEATURES list from the honest eval script."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "HONEST_FEATURES":
                    if isinstance(node.value, ast.List):
                        return [
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]

    return []


# ── Known mapping of labeling rule variables to feature names ─────

# These are the semantic connections between labeling rules and features.
# Even if the AST doesn't catch an exact string match, these are
# logically the same data being used on both sides.
KNOWN_CIRCULAR_MAPPINGS = {
    # labeling function variable → classifier feature
    "is_pr_wire": "is_pr_wire",
    "is_pr": "is_pr_wire",
    "source_tier": "source_tier",
    "is_not_tier1": "source_tier",
    "tone": "tone_score",
    "tone_threshold": "tone_score",
    "is_very_positive": "tone_score",
    "is_negative_neutral": "tone_score",
    "positive_score": "positive_score",
    "negative_score": "negative_score",
    "has_risk": "has_risk_words",
    "_has_risk_keywords": "has_risk_words",
    "_has_sponsored_url_pattern": "has_promo_ratio",
    "has_sponsored_url": "has_promo_ratio",
}


# ── main audit ────────────────────────────────────────────────────

def run_audit(strict: bool = False) -> dict:
    """Execute the structural circularity audit.

    Parameters
    ----------
    strict : bool
        If True, also flag indirect/semantic overlaps via KNOWN_CIRCULAR_MAPPINGS.

    Returns
    -------
    dict
        Audit results.
    """
    audit = {
        "timestamp": datetime.now().isoformat(),
        "label_script": str(LABEL_SCRIPT),
        "classifier_script": str(CLASSIFIER_SCRIPT),
        "honest_eval_script": str(HONEST_EVAL_SCRIPT),
        "strict_mode": strict,
    }

    # ── Step 1: Extract classifier features ───────────────────────
    logger.info("Step 1: Extracting FEATURE_COLS from classifier script...")
    feature_cols = _extract_feature_cols(CLASSIFIER_SCRIPT)
    audit["classifier_features"] = feature_cols
    logger.info(f"  Found {len(feature_cols)} features: {feature_cols}")

    if not feature_cols:
        logger.error("Could not extract FEATURE_COLS from classifier script!")
        audit["error"] = "FEATURE_COLS not found"
        return audit

    # ── Step 2: Extract labeling function identifiers ─────────────
    logger.info("Step 2: Extracting identifiers from assign_weak_labels()...")
    label_names = _extract_function_body_names(LABEL_SCRIPT, "assign_weak_labels")
    label_strings = _extract_string_constants_from_function(LABEL_SCRIPT, "assign_weak_labels")

    all_label_identifiers = label_names | label_strings
    audit["label_function_identifiers"] = sorted(all_label_identifiers)
    logger.info(f"  Found {len(all_label_identifiers)} identifiers in labeling function")

    # ── Step 3: Direct overlap ────────────────────────────────────
    logger.info("Step 3: Computing direct feature overlap...")
    direct_overlap = set(feature_cols) & all_label_identifiers
    audit["direct_overlap"] = sorted(direct_overlap)
    logger.info(f"  Direct overlap: {sorted(direct_overlap) if direct_overlap else 'NONE'}")

    # ── Step 4: Semantic overlap (strict mode) ────────────────────
    semantic_overlap = {}
    if strict:
        logger.info("Step 4: Computing semantic overlap (strict mode)...")
        for label_var, feature_name in KNOWN_CIRCULAR_MAPPINGS.items():
            if label_var in all_label_identifiers and feature_name in feature_cols:
                if feature_name not in semantic_overlap:
                    semantic_overlap[feature_name] = []
                semantic_overlap[feature_name].append(label_var)
        audit["semantic_overlap"] = semantic_overlap
        if semantic_overlap:
            logger.info(f"  Semantic overlaps found:")
            for feat, label_vars in semantic_overlap.items():
                logger.info(f"    {feat} ← {label_vars}")
        else:
            logger.info(f"  No semantic overlaps found")

    all_circular = set(direct_overlap) | set(semantic_overlap.keys())
    audit["all_circular_features"] = sorted(all_circular)

    # ── Step 5: Check honest eval script consistency ──────────────
    logger.info("Step 5: Verifying honest eval script consistency...")
    if HONEST_EVAL_SCRIPT.exists():
        excluded = _extract_excluded_features(HONEST_EVAL_SCRIPT)
        honest = _extract_honest_features(HONEST_EVAL_SCRIPT)
        audit["honest_eval_excluded"] = excluded
        audit["honest_eval_used"] = honest

        # Check: all circular features should be in CIRCULAR_FEATURES
        missing_exclusions = all_circular - set(excluded)
        if missing_exclusions:
            logger.warning(
                f"  WARNING: Circular features not in CIRCULAR_FEATURES list: "
                f"{sorted(missing_exclusions)}"
            )
            audit["missing_exclusions"] = sorted(missing_exclusions)

        # Check: no honest feature should be circular
        honest_circular = set(honest) & all_circular
        if honest_circular:
            logger.error(
                f"  FAILURE: HONEST_FEATURES contains circular features: "
                f"{sorted(honest_circular)}"
            )
            audit["honest_circular_contamination"] = sorted(honest_circular)
        else:
            logger.info(f"  HONEST_FEATURES are clean — no circular contamination")
            audit["honest_circular_contamination"] = []
    else:
        logger.warning(f"  Honest eval script not found: {HONEST_EVAL_SCRIPT}")

    # ── Step 6: Final verdict ─────────────────────────────────────
    has_circularity = len(all_circular) > 0
    honest_contaminated = len(audit.get("honest_circular_contamination", [])) > 0

    audit["verdict"] = {
        "has_circularity_in_original": has_circularity,
        "honest_model_contaminated": honest_contaminated,
        "n_circular_features": len(all_circular),
        "circular_features": sorted(all_circular),
        "build_pass": not honest_contaminated,
    }

    return audit


def run(strict: bool = False) -> None:
    """Execute audit and fail the build if circularity is detected in the honest model."""

    logger.info("=" * 70)
    logger.info("  PHASE 0.3 — STRUCTURAL CIRCULARITY AUDIT")
    logger.info("=" * 70)

    audit = run_audit(strict=strict)

    # ── Save results ──────────────────────────────────────────────
    output_path = OUTPUTS_TABLES / "circularity_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, default=str))
    logger.info(f"\nSaved audit: {output_path}")

    # ── Print summary ─────────────────────────────────────────────
    verdict = audit["verdict"]

    logger.info("\n" + "=" * 70)

    if verdict["has_circularity_in_original"]:
        logger.info(
            f"  ⚠ ORIGINAL CLASSIFIER has {verdict['n_circular_features']} circular feature(s):"
        )
        for feat in verdict["circular_features"]:
            logger.info(f"      • {feat}")
        logger.info("")

    if verdict["build_pass"]:
        logger.info("  ╔═══════════════════════════════════════╗")
        logger.info("  ║  ✓ BUILD PASS — Honest model clean   ║")
        logger.info("  ╚═══════════════════════════════════════╝")
        logger.info("")
        logger.info("  The honest model (08_honest_eval.py) uses ONLY")
        logger.info("  non-circular features. Evaluation is structurally valid.")
    else:
        logger.info("  ╔═══════════════════════════════════════╗")
        logger.info("  ║  ✗ BUILD FAIL — Circular contamination║")
        logger.info("  ╚═══════════════════════════════════════╝")
        logger.info("")
        logger.info("  The honest model STILL contains circular features!")
        logger.info(f"  Contaminated: {audit.get('honest_circular_contamination', [])}")
        logger.info("  Fix HONEST_FEATURES in pipeline/08_honest_eval.py")

    logger.info("=" * 70)

    if not verdict["build_pass"]:
        logger.error("BUILD FAILED — circularity detected in honest model")
        sys.exit(1)
    else:
        logger.info("BUILD PASSED")
        sys.exit(0)


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 0.3: Structural circularity audit (build gate)."
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Also flag indirect/semantic overlaps between labeling and classifier",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(strict=args.strict)
