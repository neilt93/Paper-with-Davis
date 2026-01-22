"""CLI script to run visibility evaluation on a local CSV/Excel file.

Usage: see README or run with -h. This is a thin wrapper around the
core logic in `visibility_eval_core.run_on_dataframe`.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure the `scripts` directory (where visibility_eval_core.py lives) is on sys.path
SCRIPT_ROOT = Path(__file__).resolve().parents[1]  # .../scripts
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

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
        help="Model name: e.g. gpt, gemini, llava-onevision, qwen-vl, qwen-vl-4bit, or a HF repo id for local.",
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
            "(<model>_I0q0_json / <model>_I0q1_json / <model>_I1q0_json / <model>_I1q1_json) are empty."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Determine output path early so we can use it for incremental saving
    out_path = args.out
    if out_path is None:
        base, ext = os.path.splitext(args.input)
        out_path = f"{base}.{args.model}.vlm.csv"

    # If output file exists and we're using --only-missing, load it to preserve existing results
    if args.only_missing and os.path.exists(out_path):
        try:
            print(f"[INFO] Loading existing output file to preserve progress: {out_path}", file=sys.stderr)
            df_output = load_table(out_path)
            df_input = load_table(args.input)
            
            # Use input as base, but preserve output columns from existing file
            # Merge on 'id' column if it exists, otherwise use index
            df = df_input.copy()
            if "id" in df_output.columns and "id" in df.columns:
                # Merge on id to preserve existing outputs
                output_cols = [col for col in df_output.columns if col not in df.columns]
                if output_cols:
                    df_output_subset = df_output[["id"] + output_cols].set_index("id")
                    df = df.set_index("id")
                    for col in output_cols:
                        if col in df_output_subset.columns:
                            df[col] = df_output_subset[col]
                    df = df.reset_index()
            else:
                # Fallback: preserve output columns by position (less reliable)
                output_cols = [col for col in df_output.columns if col not in df.columns]
                for col in output_cols:
                    if col in df_output.columns and len(df_output) == len(df):
                        df[col] = df_output[col].values
        except Exception as e:
            print(f"[WARN] Failed to load existing output file '{out_path}': {e}. Starting fresh.", file=sys.stderr)
            df = load_table(args.input)
    else:
        try:
            df = load_table(args.input)
        except Exception as e:
            print(f"[ERROR] Failed to read table '{args.input}': {e}", file=sys.stderr)
            sys.exit(1)

    try:
        # Pass save_path for incremental saving after each row
        df_done = run_on_dataframe(df, args.model, only_missing=args.only_missing, save_path=out_path)
    except KeyboardInterrupt:
        print(f"\n[INFO] Interrupted by user. Progress saved to: {out_path}", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Core evaluation failed: {e}", file=sys.stderr)
        print(f"[INFO] Partial progress may be saved to: {out_path}", file=sys.stderr)
        sys.exit(1)

    # Final save (in case incremental saves missed anything)
    try:
        out_ext = os.path.splitext(out_path.lower())[1]
        if out_ext in {".xlsx", ".xls"}:
            df_done.to_excel(out_path, index=False)
        else:
            df_done.to_csv(out_path, index=False)
    except Exception as e:
        print(f"[ERROR] Failed to write final output table '{out_path}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Done. Final save: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()


