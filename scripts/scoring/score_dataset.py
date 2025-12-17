"""CLI to score a visibility-eval CSV according to the paper's metrics.

This is intentionally generic: you point it at a CSV that already contains
gold labels, predicted labels, confidences, and (optionally) variant and
ToM metadata. Column names are configurable via flags.

Typical workflow for a given model:
  1) Run the model locally to produce <model>_base_json / <model>_flip_json.
  2) Use a small prep script (see `prepare_scored_csv.py`) to parse those JSON
     fields into gold_label / pred_label / confidence columns.
  3) Call this script on the resulting *.scored.csv file.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Allow running as a standalone script: add this folder to sys.path
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from metrics import compute_all_metrics, composite_score_with_effective_weights  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a visibility-eval CSV with confidence-aware metrics."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input CSV containing gold/pred/confidence columns.",
    )
    parser.add_argument(
        "--gold-col",
        default="gold_label",
        help="Column with gold labels (default: gold_label).",
    )
    parser.add_argument(
        "--pred-col",
        default="pred_label",
        help="Column with predicted labels (default: pred_label).",
    )
    parser.add_argument(
        "--conf-col",
        default="confidence",
        help="Column with prediction confidences in [0,1] (default: confidence).",
    )
    parser.add_argument(
        "--family-col",
        default="family_id",
        help="Column grouping base/variant items (default: family_id).",
    )
    parser.add_argument(
        "--variant-col",
        default="variant_relation",
        help=(
            "Column with variant_relation ∈ {BASE, TEXT_FLIP, IMAGE_FLIP, DOUBLE_FLIP, INVARIANT} "
            "(default: variant_relation)."
        ),
    )
    parser.add_argument(
        "--tom-flag-col",
        default="is_second_order_tom",
        help="Boolean/0-1 column indicating second-order ToM items (default: is_second_order_tom).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input CSV not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV '{args.input}': {e}", file=sys.stderr)
        sys.exit(1)

    # If no explicit ToM flag exists, derive it from family_id prefix (MA-*) when possible.
    if args.tom_flag_col not in df.columns and args.family_col in df.columns:
        df[args.tom_flag_col] = df[args.family_col].astype(str).str.match(r"^MA-").astype(int)

    try:
        metrics = compute_all_metrics(
            df,
            gold_col=args.gold_col,
            pred_col=args.pred_col,
            confidence_col=args.conf_col,
            family_col=args.family_col,
            relation_col=args.variant_col,
            tom_flag_col=args.tom_flag_col,
        )
        final, eff_w = composite_score_with_effective_weights(metrics)
    except Exception as e:
        print(f"[ERROR] Failed to compute metrics: {e}", file=sys.stderr)
        sys.exit(1)

    # Pretty-print a small report.
    print("=== Visibility Evaluation Scores ===")
    print(f"Input file              : {args.input}")
    print()
    print("Component metrics:")
    print(f"  Core confidence-aware accuracy (CAA, alpha=0.25) : {metrics['core_confidence_accuracy']:.4f}")
    print(f"  Image flip MEFR (I_MEFR | BASE-correct)   : {metrics['I_MEFR']:.4f}")
    print(f"  Text flip MEFR  (T_MEFR | BASE-correct)   : {metrics['T_MEFR']:.4f}")
    print(f"  Mean conditional flip rate (MEFR)         : {metrics['MEFR']:.4f}")
    print(f"  Double flip accuracy (DFAcc | BASE-correct): {metrics['DFAcc']:.4f}")
    print(f"  Abstention quality (AURC-normalized)      : {metrics['abstention_quality']:.4f}")
    print(f"  Second-order ToM accuracy                 : {metrics['second_order_accuracy']:.4f}")
    print(f"  Strict invariance consistency (IC_strict) : {metrics['ic_strict']:.4f}")
    print()
    print("Effective composite weights used (after NaN renormalization):")
    print(f"  w_CAA   : {eff_w['core_confidence_accuracy']:.3f}")
    print(f"  w_MEFR  : {eff_w['MEFR']:.3f}")
    print(f"  w_AURC  : {eff_w['abstention_quality']:.3f}")
    print(f"  w_ToM   : {eff_w['second_order_accuracy']:.3f}")
    print()
    print(f"Final composite score            : {final:.4f}")


if __name__ == "__main__":
    main()


