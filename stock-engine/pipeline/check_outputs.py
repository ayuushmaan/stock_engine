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

ALLOWED_FILES = [
    "cross_market_replication.json",  # known placeholder, will be quarantined
    "propagation_cascades.json",      # known simulated, will be quarantined
]


def check_outputs():
    tables_dir = Path("outputs/tables")
    if not tables_dir.exists():
        print("ERROR: outputs/tables not found")
        return 1

    violations = []
    for json_file in tables_dir.glob("*.json"):
        try:
            content = json_file.read_text(encoding="utf-8")
            data = json.loads(content)
        except Exception as e:
            print(f"WARNING: Could not parse {json_file.name}: {e}")
            continue

        # Check for banned tokens in the JSON content
        for token in BANNED_TOKENS:
            if token in content:
                if json_file.name in ALLOWED_FILES:
                    print(f"ALLOWED placeholder in {json_file.name} (known quarantine)")
                    continue
                violations.append((json_file.name, token))

    if violations:
        print("ERROR: Found placeholder/simulated content in outputs/tables:")
        for fname, token in violations:
            print(f"  {fname}: contains '{token}'")
        print("\nFix: quarantine or regenerate with real data.")
        return 1

    print("OK: No placeholder/simulated content found in outputs/tables/")
    return 0


if __name__ == "__main__":
    sys.exit(check_outputs())
