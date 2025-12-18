"""Shared CLI helper for per-model runner scripts.

These wrappers exist so you can run a model with a stable alias (clean output
column prefixes) without remembering the exact `--model` string.

Each wrapper passes a fixed `model_name` into `visibility_eval_core.run_on_dataframe`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure the `scripts` directory (where visibility_eval_core.py lives) is on sys.path
SCRIPT_ROOT = Path(__file__).resolve().parents[1]  # .../scripts
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from visibility_eval_core import load_table, run_on_dataframe  # noqa: E402


def main_with_model(model_name: str, description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", "-i", required=True, help="Input CSV/Excel with visibility dataset.")
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
        raise SystemExit(1)

    try:
        df = load_table(args.input)
    except Exception as e:
        print(f"[ERROR] Failed to read table '{args.input}': {e}", file=sys.stderr)
        raise SystemExit(1)

    try:
        df_done = run_on_dataframe(df, model_name, only_missing=bool(args.only_missing))
    except Exception as e:
        msg = str(e)
        # Common HF gated-model error message.
        if "Cannot access gated repo" in msg or "401 Client Error" in msg or "restricted" in msg:
            print(
                "[HINT] This model is gated on Hugging Face. Run `huggingface-cli login` "
                "or set HF_TOKEN in your environment, then retry.",
                file=sys.stderr,
            )
        print(f"[ERROR] Core evaluation failed: {e}", file=sys.stderr)
        raise SystemExit(1)

    out_path = args.out
    if out_path is None:
        base, _ext = os.path.splitext(args.input)
        out_path = f"{base}.{model_name}.vlm.csv"

    try:
        out_ext = os.path.splitext(out_path.lower())[1]
        if out_ext in {".xlsx", ".xls"}:
            df_done.to_excel(out_path, index=False)
        else:
            df_done.to_csv(out_path, index=False)
    except Exception as e:
        print(f"[ERROR] Failed to write output table '{out_path}': {e}", file=sys.stderr)
        raise SystemExit(1)

    print(f"[INFO] Done. saved={out_path}", file=sys.stderr)


