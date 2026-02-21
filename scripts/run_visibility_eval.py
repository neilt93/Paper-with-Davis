#!/usr/bin/env python3
"""CLI entry point for running VLM visibility evaluation."""

import argparse
import os
import sys
from pathlib import Path

# Ensure the scripts directory is on the path
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from visibility_eval_core import load_table, run_on_dataframe


def main():
    parser = argparse.ArgumentParser(description="Run VLM visibility evaluation")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV/Excel")
    parser.add_argument("--model", "-m", required=True, help="Model name (gpt, gemini, claude, etc.)")
    parser.add_argument("--only-missing", action="store_true", help="Skip rows that already have outputs")
    parser.add_argument("--out", "-o", default=None, help="Output path (default: <input>.<model>.vlm.csv)")
    args = parser.parse_args()

    df = load_table(args.input)

    out_path = args.out
    if out_path is None:
        base, ext = os.path.splitext(args.input)
        out_path = f"{base}.{args.model}.vlm.csv"

    # If resuming, load existing output to preserve prior results
    if args.only_missing and os.path.exists(out_path):
        import pandas as pd
        df = pd.read_csv(out_path)
        print(f"[INFO] Resuming from existing output: {out_path}", file=sys.stderr)

    result = run_on_dataframe(df, args.model, only_missing=args.only_missing, save_path=out_path)
    result.to_csv(out_path, index=False)
    print(f"[DONE] Output saved to: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
