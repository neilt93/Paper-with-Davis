"""CLI script to run visibility evaluation on a local CSV/Excel file.

Usage: see README or run with -h. This is a thin wrapper around the
core logic in `visibility_eval_core.run_on_dataframe`.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root (where visibility_eval_core.py lives) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visibility_eval_core import load_table, run_on_dataframe


def main():
    parser = argparse.ArgumentParser(
        description="Run visibility eval for a CSV/Excel against a VLM (API or local)."
    )
    parser.add_argument("--input", "-i", required=True, help="Input CSV/Excel with visibility dataset.")
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name: e.g. gpt, gemini, llava-onevision, qwen-vl, or a HF repo id for local.",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Output table path (default: <input_basename>.<model>.vlm.csv)",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help=(
            "Only run the model for rows where the output columns for this model "
            "(<model>_base_json / <model>_flip_json) are empty."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        df = load_table(args.input)
    except Exception as e:
        print(f"[ERROR] Failed to read table '{args.input}': {e}", file=sys.stderr)
        sys.exit(1)

    try:
        df_done = run_on_dataframe(df, args.model, only_missing=args.only_missing)
    except Exception as e:
        print(f"[ERROR] Core evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)

    out_path = args.out
    if out_path is None:
        base, ext = os.path.splitext(args.input)
        out_path = f"{base}.{args.model}.vlm.csv"

    try:
        out_ext = os.path.splitext(out_path.lower())[1]
        if out_ext in {".xlsx", ".xls"}:
            df_done.to_excel(out_path, index=False)
        else:
            df_done.to_csv(out_path, index=False)
    except Exception as e:
        print(f"[ERROR] Failed to write output table '{out_path}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Done. saved={out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()


