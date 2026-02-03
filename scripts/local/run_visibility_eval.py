"""CLI script to run visibility evaluation on a local CSV/Excel file.

Usage: see README or run with -h. This is a thin wrapper around the
core logic in `visibility_eval_core.run_on_dataframe`.
"""

import argparse
import os
import sys
from pathlib import Path

# --- Fix 1: Robust sys.path logic that searches for visibility_eval_core.py ---
THIS_FILE = Path(__file__).resolve()


def _add_core_dir_to_syspath():
    """Search up from this script for visibility_eval_core.py and add to sys.path."""
    for p in [THIS_FILE.parent, *THIS_FILE.parents]:
        if (p / "visibility_eval_core.py").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return
    raise ImportError(
        f"Could not locate visibility_eval_core.py near {THIS_FILE}. "
        "Make sure it is in the same directory as this script or a parent directory."
    )


_add_core_dir_to_syspath()

from visibility_eval_core import load_table, run_on_dataframe


# --- Fix 2: Sanitize model names for safe filenames (no slashes) ---
def _safe_tag(s: str) -> str:
    """Convert model name to filesystem-safe tag (no slashes, spaces, etc.)."""
    s = s.strip()
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in s).strip("_")


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

    # Sanitize model name for filenames (HF repo IDs like "Qwen/Qwen2-VL" have slashes)
    model_tag = _safe_tag(args.model)

    # Determine output path early so we can use it for incremental saving
    out_path = args.out
    if out_path is None:
        base, _ = os.path.splitext(args.input)
        out_path = f"{base}.{model_tag}.vlm.csv"

    # --- Fix 3: Create output directory if it doesn't exist ---
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # If output file exists and we're using --only-missing, load it to preserve existing results
    if args.only_missing and os.path.exists(out_path):
        try:
            print(f"[INFO] Loading existing output file to preserve progress: {out_path}", file=sys.stderr)
            df_output = load_table(out_path)
            df_input = load_table(args.input)

            # --- Fix 4: Robust merge that handles overlapping columns ---
            df = df_input.copy()

            if "id" in df.columns and "id" in df_output.columns:
                # Merge on id, using suffixes to handle overlapping columns
                df = df.merge(df_output, on="id", how="left", suffixes=("", "__old"))

                for col in df_output.columns:
                    if col == "id":
                        continue
                    old_col = f"{col}__old"
                    if old_col in df.columns:
                        if col in df.columns:
                            # Fill empty/NaN values in df[col] from the old output
                            df[col] = df[col].where(
                                df[col].notna() & (df[col] != ""),
                                df[old_col]
                            )
                        else:
                            df[col] = df[old_col]
                        df.drop(columns=[old_col], inplace=True)
            else:
                # Fallback by position if lengths match (less reliable)
                if len(df_output) == len(df):
                    for col in df_output.columns:
                        if col not in df.columns:
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
