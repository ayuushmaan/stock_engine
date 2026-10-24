"""Pipeline runner for Human & Multi-Judge Validation (Step 10)."""
import argparse
from pathlib import Path
from pipeline.human_validation import run_human_validation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inter-annotator agreement evaluation")
    parser.add_argument("--input", type=str, default=None, help="Path to labeled CSV")
    args = parser.parse_args()

    in_p = Path(args.input) if args.input else None
    run_human_validation(in_p)
