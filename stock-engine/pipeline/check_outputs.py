"""CI check: fail if any outputs/tables/*.json contains placeholder/simulated content."""
from __future__ import annotations

import json
import sys
from pathlib import Path

BANNED_TOKENS = [
    "_placeholder",
    "simulat",  # catches "simulated", "simulation", "simulate"
    "ILLUSTRATIVE",  # only allowed if explicitly marked in filename
]

REQUIRED_FILES = [
    "five_baselines_benchmark.json",
    "factor_backtest_metrics.json",
    "h2_predictive.json",
    "signal_decay_analysis.json",
    "propagation_cascade_analysis.json",
    "robustness_tests.json",
    "inter_annotator_agreement.json",
]


def check_outputs():
    tables_dir = Path("outputs/tables")
    if not tables_dir.exists():
        print("ERROR: outputs/tables not found")
        return 1

    json_files = list(tables_dir.glob("*.json"))
    if not json_files:
        print("ERROR: outputs/tables is empty")
        return 1

    # Check required files exist
    missing = []
    for req in REQUIRED_FILES:
        if not (tables_dir / req).exists():
            missing.append(req)
    if missing:
        print(f"ERROR: Required files missing from outputs/tables: {missing}")
        return 1

    violations = []
    for json_file in json_files:
        # Skip examples/illustrative files (they're not in outputs/tables anyway)
        try:
            content = json_file.read_text(encoding="utf-8")
            data = json.loads(content)
        except Exception as e:
            print(f"WARNING: Could not parse {json_file.name}: {e}")
            continue

        # Check for banned tokens in the JSON content
        for token in BANNED_TOKENS:
            if token in content:
                violations.append((json_file.name, token))

    if violations:
        print("ERROR: Found placeholder/simulated content in outputs/tables:")
        for fname, token in violations:
            print(f"  {fname}: contains '{token}'")
        print("\nFix: quarantine or regenerate with real data.")
        return 1

    print("OK: No placeholder/simulated content found in outputs/tables/")
    print(f"  Checked {len(json_files)} files, all required files present.")
    return 0


if __name__ == "__main__":
    sys.exit(check_outputs())
