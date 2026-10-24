#!/usr/bin/env python
"""
One-command pipeline runner for NIFTY 50 News-Signal Research Engine.

Usage:
    python run_all.py --dry-run          # Quick test (3 tickers, 1 month GDELT, 10k articles)
    python run_all.py --full             # Full pipeline (all tickers, 2020-2026)
    python run_all.py --steps 01 02 03   # Specific steps only
    python run_all.py --test             # Run tests + circularity check
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
PYTHON = sys.executable

PIPELINE_STEPS = [
    ("01_fetch_prices", "Fetch NIFTY 50 prices & compute returns"),
    ("02_fetch_gdelt", "Fetch GDELT 2.0 GKG from BigQuery"),
    ("03_label_news", "Label articles: time buckets, effective dates, weak labels"),
    ("04_sponsored_classifier", "Train LightGBM classifier & score all articles"),
    ("05_aggregate_signals", "Aggregate to daily stock-level signals (2x2 cells)"),
]

CI_CHECKS = [
    ("check_circularity", "AST circularity guard (CI gate)"),
    ("tests", "Unit test suite (22 tests)"),
]

RESEARCH_STEPS = [
    ("06_baselines", "5-baseline quant benchmark table"),
    ("08_backtest", "Predictive factor backtest"),
]


def run_cmd(cmd: list[str], cwd: Path = None, env: dict = None) -> int:
    """Run command, stream output, return exit code."""
    print(f"\n{'='*60}")
    print(f">>> {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, env=env)
    return result.returncode


def run_pipeline_step(step: str, dry_run: bool = False) -> int:
    """Run a single pipeline step."""
    script = PROJECT_ROOT / "pipeline" / f"{step}.py"
    if not script.exists():
        print(f"[FAIL] Script not found: {script}")
        return 1
    cmd = [PYTHON, str(script)]
    if dry_run:
        cmd.append("--dry-run")
    return run_cmd(cmd)


def run_research_step(step: str, dry_run: bool = False) -> int:
    """Run a research script."""
    script = PROJECT_ROOT / "research" / f"{step}.py"
    if not script.exists():
        print(f"[FAIL] Script not found: {script}")
        return 1
    cmd = [PYTHON, str(script)]
    if dry_run:
        cmd.append("--dry-run")
    return run_cmd(cmd)


def run_ci_checks() -> int:
    """Run circularity check + tests."""
    failed = 0
    
    # Circularity check
    rc = run_cmd([PYTHON, str(PROJECT_ROOT / "pipeline" / "check_circularity.py")])
    if rc != 0:
        failed += 1
    
    # Tests
    rc = run_cmd([PYTHON, "-m", "pytest", "tests/", "-v"])
    if rc != 0:
        failed += 1
    
    return 0 if failed == 0 else 1


def run_dashboard() -> int:
    """Launch Streamlit dashboard."""
    return run_cmd([PYTHON, "-m", "streamlit", "run", "app/app.py"])


def main():
    parser = argparse.ArgumentParser(description="NIFTY 50 News-Signal Pipeline Runner")
    parser.add_argument("--dry-run", action="store_true", help="Run with small data subsets")
    parser.add_argument("--full", action="store_true", help="Run full pipeline (all data)")
    parser.add_argument("--steps", nargs="+", help="Specific pipeline steps to run (e.g., 01_fetch_prices 02_fetch_gdelt)")
    parser.add_argument("--research", nargs="+", help="Research steps to run (e.g., 06_baselines 08_backtest)")
    parser.add_argument("--test", action="store_true", help="Run CI checks (circularity + tests)")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--all", action="store_true", help="Run everything: pipeline + research + tests")
    
    args = parser.parse_args()
    
    # Default to dry-run if no mode specified
    if not any([args.dry_run, args.full, args.steps, args.research, args.test, args.dashboard, args.all]):
        args.dry_run = True
        print("No mode specified → defaulting to --dry-run")
    
    failed = 0
    
    # Pipeline steps
    if args.all or args.full or args.dry_run or args.steps:
        steps_to_run = args.steps if args.steps else [s[0] for s in PIPELINE_STEPS]
        print(f"\n{'#'*60}")
        print(f"# PIPELINE: {len(steps_to_run)} steps {'(DRY-RUN)' if args.dry_run else '(FULL)'}")
        print(f"{'#'*60}")
        for step in steps_to_run:
            rc = run_pipeline_step(step, dry_run=args.dry_run)
            if rc != 0:
                print(f"[FAIL] Step {step} FAILED (exit code {rc})")
                failed += 1
            else:
                print(f"[OK] Step {step} COMPLETE")
    
    # Research steps
    if args.all or args.research:
        research_to_run = args.research if args.research else [s[0] for s in RESEARCH_STEPS]
        print(f"\n{'#'*60}")
        print(f"# RESEARCH: {len(research_to_run)} steps")
        print(f"{'#'*60}")
        for step in research_to_run:
            rc = run_research_step(step, dry_run=args.dry_run)
            if rc != 0:
                print(f"[FAIL] Research {step} FAILED (exit code {rc})")
                failed += 1
            else:
                print(f"[OK] Research {step} COMPLETE")
    
    # CI checks
    if args.all or args.test:
        print(f"\n{'#'*60}")
        print(f"# CI CHECKS")
        print(f"{'#'*60}")
        rc = run_ci_checks()
        if rc != 0:
            failed += 1
    
    # Dashboard
    if args.dashboard:
        run_dashboard()
        return 0
    
    # Summary
    print(f"\n{'#'*60}")
    if failed == 0:
        print("# [OK] ALL TASKS COMPLETED SUCCESSFULLY")
    else:
        print(f"# [FAIL] {failed} TASK(S) FAILED")
    print(f"{'#'*60}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())