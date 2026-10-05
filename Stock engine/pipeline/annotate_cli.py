r"""Pipeline Interactive Tool — Rapid Hand-Labeling CLI for Phase 0.

An interactive console tool to make labeling 1,000 articles fast, seamless,
and frictionless. Automatically tracks progress, opens URLs in your default
browser, accepts single-key inputs, auto-saves on every action, and resumes
where you left off.

Features:
    - Auto-opens URL in browser (toggleable)
    - Single-key shortcuts: '1'/'s' for Sponsored, '0'/'o' for Organic, '-1'/'u' for Unclear
    - Fast reason templates or custom input
    - Jump to any sample ID / Go back to fix previous mistakes
    - Real-time progress stats & class breakdown
    - Zero data loss (saves CSV on every annotation)

Usage:
    python pipeline/annotate_cli.py
    python pipeline/annotate_cli.py --no-browser   # don't auto-open URLs
    python pipeline/annotate_cli.py --start-id 150 # jump directly to sample #150
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import pandas as pd

# ── project imports ───────────────────────────────────────────────
from config.settings import DATA_FINAL, setup_logging

logger = setup_logging()

CSV_PATH = DATA_FINAL / "hand_labels_workbook.csv"

COMMON_REASONS = {
    "1": [
        "1. Direct PR wire release / corporate statement",
        "2. Advertorial / brand studio / promotional partner content",
        "3. Product/service launch with purely hyperbolic praise",
        "4. Award/CSR announcement with zero financial or risk context",
        "5. Other sponsored / promotional rationale",
    ],
    "0": [
        "1. Genuine editorial reporting / investigative journalism",
        "2. Earnings analysis with balanced metrics & risk factors",
        "3. Regulatory scrutiny / legal / governance reporting",
        "4. Industry-wide market commentary with balanced tone",
        "5. Other organic editorial rationale",
    ],
    "-1": [
        "1. 404 / Broken link / Paywall blocked",
        "2. Ambiguous syndication / Mixed editorial & promo",
        "3. Non-English / Parsing failure",
        "4. Other unclear",
    ],
}


def print_header(sample_id: int, total: int, row: pd.Series, df: pd.DataFrame) -> None:
    labeled = df["hand_label"].dropna()
    valid_labeled = labeled[labeled.isin([0, 1, -1, "0", "1", "-1", 0.0, 1.0, -1.0])]
    n_done = len(valid_labeled)
    pct_done = 100.0 * n_done / total if total > 0 else 0

    spon_count = (pd.to_numeric(df["hand_label"], errors="coerce") == 1).sum()
    org_count = (pd.to_numeric(df["hand_label"], errors="coerce") == 0).sum()
    unc_count = (pd.to_numeric(df["hand_label"], errors="coerce") == -1).sum()

    print("\n" + "=" * 80)
    print(f"  SAMPLE #{sample_id} / {total}   |   Progress: {n_done}/{total} ({pct_done:.1f}%)")
    print(f"  Stats: Sponsored (1): {spon_count}  |  Organic (0): {org_count}  |  Unclear (-1): {unc_count}")
    print("=" * 80)

    url = str(row.get("source_url", ""))
    domain = str(row.get("source_domain", ""))
    tier = row.get("source_tier", "N/A")
    tone = row.get("tone", 0.0)
    pos = row.get("positive_score", 0.0)
    neg = row.get("negative_score", 0.0)
    wc = row.get("word_count", 0.0)
    year = row.get("year", "N/A")
    ticker = row.get("ticker", "N/A")
    cur_label = row.get("hand_label", "")
    cur_reason = row.get("reason", "")

    print(f"  Domain : {domain} (Tier {tier})   |   Year: {year}   |   Ticker: {ticker}")
    print(f"  Tone   : {tone:+.2f}  (Pos: {pos:.2f}, Neg: {neg:.2f})   |   Word Count: {wc:.0f}")
    print(f"  URL    : {url}")
    if pd.notna(cur_label) and str(cur_label).strip() != "":
        lbl_str = {1: "SPONSORED (1)", 0: "ORGANIC (0)", -1: "UNCLEAR (-1)"}.get(
            int(float(cur_label)), str(cur_label)
        )
        print(f"  [Current Annotation]: {lbl_str} — \"{cur_reason}\"")
    print("-" * 80)


def choose_reason(label_val: int) -> str:
    key = str(label_val)
    options = COMMON_REASONS.get(key, [])
    print("\nSelect a reason template (or type custom text, or press Enter to skip):")
    for opt in options:
        print(f"  {opt}")
    choice = input("Reason (1-5 or custom text): ").strip()

    if choice == "":
        return "Manual review"
    if choice in ["1", "2", "3", "4", "5"] and int(choice) <= len(options):
        # Extract text after number
        raw = options[int(choice) - 1]
        return raw.split(". ", 1)[-1]
    return choice


def run_interactive(csv_path: Path, auto_open: bool = True, start_id: int | None = None) -> None:
    if not csv_path.exists():
        logger.error(f"Workbook not found at {csv_path}. Run 07_sample_for_hand_labels.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    total_rows = len(df)
    logger.info(f"Loaded {total_rows} articles for labeling.")

    # Find starting index
    if start_id is not None:
        curr_idx = max(0, min(start_id - 1, total_rows - 1))
    else:
        # Find first unlabeled row
        unlabeled_mask = (
            df["hand_label"].isna()
            | (df["hand_label"].astype(str).str.strip() == "")
        )
        unlabeled_indices = df.index[unlabeled_mask].tolist()
        curr_idx = unlabeled_indices[0] if unlabeled_indices else 0

    print("\n" + "#" * 80)
    print("  NIFTY 50 NEWS CREDIBILITY — RAPID HAND-LABELING CLI")
    print("  Shortcuts:")
    print("    [1] or [s]  : Label SPONSORED (1)")
    print("    [0] or [o]  : Label ORGANIC (0)")
    print("    [-1] or [u] : Label UNCLEAR (-1)")
    print("    [o]         : Open URL in browser")
    print("    [b]         : Go Back to previous item")
    print("    [j <id>]    : Jump to sample ID")
    print("    [q]         : Save and Exit")
    print("#" * 80)

    try:
        while 0 <= curr_idx < total_rows:
            row = df.iloc[curr_idx]
            sample_id = curr_idx + 1
            url = str(row.get("source_url", ""))

            print_header(sample_id, total_rows, row, df)

            if auto_open and url.startswith("http"):
                try:
                    webbrowser.open_new_tab(url)
                except Exception as e:
                    print(f"  [Notice] Could not auto-open browser: {e}")

            while True:
                try:
                    cmd = input("Command [1=Spon / 0=Org / -1=Unclear / o=Open / b=Back / q=Quit]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    cmd = "q"

                if cmd in ["q", "quit", "exit"]:
                    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    print(f"\nProgress saved to {csv_path}. Exiting...")
                    return

                if cmd in ["o", "open"]:
                    if url.startswith("http"):
                        webbrowser.open_new_tab(url)
                    else:
                        print("  Invalid URL.")
                    continue

                if cmd in ["b", "back"]:
                    curr_idx = max(0, curr_idx - 1)
                    break

                if cmd.startswith("j"):
                    parts = cmd.split()
                    if len(parts) > 1 and parts[1].isdigit():
                        target = int(parts[1]) - 1
                        if 0 <= target < total_rows:
                            curr_idx = target
                            break
                        else:
                            print(f"  Sample ID must be between 1 and {total_rows}")
                    else:
                        print("  Usage: j <sample_id> (e.g. j 45)")
                    continue

                if cmd in ["1", "s", "sponsored"]:
                    reason = choose_reason(1)
                    df.at[curr_idx, "hand_label"] = 1
                    df.at[curr_idx, "reason"] = reason
                    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    print(f"  -> Marked as SPONSORED (1): {reason}")
                    curr_idx += 1
                    break

                if cmd in ["0", "org", "organic", "g"]:
                    reason = choose_reason(0)
                    df.at[curr_idx, "hand_label"] = 0
                    df.at[curr_idx, "reason"] = reason
                    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    print(f"  -> Marked as ORGANIC (0): {reason}")
                    curr_idx += 1
                    break

                if cmd in ["-1", "u", "unclear"]:
                    reason = choose_reason(-1)
                    df.at[curr_idx, "hand_label"] = -1
                    df.at[curr_idx, "reason"] = reason
                    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    print(f"  -> Marked as UNCLEAR (-1): {reason}")
                    curr_idx += 1
                    break

                print("  Unknown command. Use 1 (sponsored), 0 (organic), -1 (unclear), o (open), b (back), or q (quit).")

    except (KeyboardInterrupt, SystemExit):
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n[Interrupted] Progress successfully saved to {csv_path}. Exiting...")
        return

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 80)
    print("  ALL 1,000 ARTICLES ANNOTATED! CONGRATULATIONS!")
    print(f"  Saved to: {csv_path}")
    print("  Now run: python pipeline/08_honest_eval.py")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rapid interactive labeling CLI.")
    parser.add_argument("--csv", type=str, default=str(CSV_PATH), help="Path to hand_labels_workbook.csv")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open URLs in browser")
    parser.add_argument("--start-id", type=int, default=None, help="Sample ID to start at (1-indexed)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_interactive(
        csv_path=Path(args.csv),
        auto_open=not args.no_browser,
        start_id=args.start_id,
    )
