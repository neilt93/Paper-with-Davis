"""
Annotate a wide XOR 2×2 VLM output CSV with overall summary metrics.

Input: a <sheet>.<model>.vlm.csv produced by scripts/local/run_visibility_eval.py
       containing the four XOR columns:
         <model>_I0q0_json, <model>_I0q1_json, <model>_I1q0_json, <model>_I1q1_json

Output: a wide CSV with the original rows/columns plus constant "overall_*" columns
        repeated on every row so the score is visible in Google Sheets.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# Make it easy to import sibling scoring modules when run as a script.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from metrics import compute_all_metrics, composite_score_with_effective_weights  # type: ignore
from score_vlm_run import extract_json_fields, _normalise_label, _normalise_confidence  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate a wide XOR VLM CSV with overall metrics for easy viewing in Sheets."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to <sheet>.<model>.vlm.csv")
    parser.add_argument("--model", "-m", required=True, help="Model prefix used in column names (e.g. gpt).")
    parser.add_argument("--out", "-o", default=None, help="Output path (default: <input>.with_scores.csv)")
    parser.add_argument(
        "--strict-3cell",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Same semantics as score_vlm_run.py: strict MEFR denominators require BASE/TEXT_FLIP/IMAGE_FLIP parsable.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    model = args.model

    # Work on Done rows only for metrics.
    done_df = df.copy()
    if "Status" in done_df.columns:
        done_df = done_df[done_df["Status"].astype(str).str.strip() == "Done"]

    if done_df.empty:
        print("[ERROR] No Status==Done rows to score.", file=sys.stderr)
        sys.exit(1)

    # Ensure we have ID
    if "id" not in done_df.columns and "ID" in done_df.columns:
        done_df["id"] = done_df["ID"]
    if "id" not in done_df.columns:
        print("[ERROR] Missing ID/id column.", file=sys.stderr)
        sys.exit(1)

    col_map = {
        f"{model}_I0q0_json": "BASE",
        f"{model}_I0q1_json": "TEXT_FLIP",
        f"{model}_I1q0_json": "IMAGE_FLIP",
        f"{model}_I1q1_json": "DOUBLE_FLIP",
    }
    missing = [c for c in col_map.keys() if c not in done_df.columns]
    if missing:
        print(f"[ERROR] Missing XOR columns: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    def xor_gold_from_base(base_label: Any) -> Optional[Dict[str, str]]:
        b = _normalise_label(base_label)
        if b == "NOT_VISIBLE":
            return {"BASE": "NOT_VISIBLE", "TEXT_FLIP": "VISIBLE", "IMAGE_FLIP": "VISIBLE", "DOUBLE_FLIP": "NOT_VISIBLE"}
        if b == "VISIBLE":
            return {"BASE": "VISIBLE", "TEXT_FLIP": "NOT_VISIBLE", "IMAGE_FLIP": "NOT_VISIBLE", "DOUBLE_FLIP": "VISIBLE"}
        return None

    # Explode into long items.
    rows_long: list[dict[str, Any]] = []
    unparsable_items = 0
    families_with_any_unparsable: set[str] = set()
    dropped_families_bad_gold = 0
    complete_headline_families: set[str] = set()

    for _, row in done_df.iterrows():
        fam = str(row["id"])
        is_tom = 1 if str(fam).startswith("MA-") else 0

        parsed_labels: Dict[str, Optional[str]] = {}
        parsed_confs: Dict[str, Optional[float]] = {}
        for col_name, rel in col_map.items():
            pred, conf, _reason = extract_json_fields(row.get(col_name))
            pred_n = _normalise_label(pred)
            conf_n = _normalise_confidence(conf)
            parsed_labels[rel] = pred_n
            parsed_confs[rel] = conf_n
            if pred_n is None:
                unparsable_items += 1
                families_with_any_unparsable.add(fam)

        golds = xor_gold_from_base(row.get("base_label"))
        if golds is None:
            dropped_families_bad_gold += 1
            continue

        if all(parsed_labels.get(rel) is not None for rel in ("BASE", "TEXT_FLIP", "IMAGE_FLIP")):
            complete_headline_families.add(fam)

        for rel in col_map.values():
            pred_n = parsed_labels[rel]
            conf_n = parsed_confs[rel]
            if pred_n is None:
                continue
            rows_long.append(
                {
                    "family_id": fam,
                    "variant_relation": rel,
                    "is_second_order_tom": is_tom,
                    "gold_label": golds[rel],
                    "pred_label": pred_n,
                    "confidence": conf_n,
                }
            )

    items_df = pd.DataFrame(rows_long)
    headline_items_df = items_df[items_df["variant_relation"].isin(["BASE", "TEXT_FLIP", "IMAGE_FLIP"])].copy()

    metrics_df = headline_items_df.copy()
    metrics_df = metrics_df[metrics_df["gold_label"].notna() & metrics_df["pred_label"].notna()]
    answered_mask = metrics_df["pred_label"].astype(str).str.upper().isin(["VISIBLE", "NOT_VISIBLE"])
    metrics_df = metrics_df[~answered_mask | metrics_df["confidence"].notna()]

    strict = bool(args.strict_3cell)
    if strict:
        eligible_family_set = complete_headline_families
    else:
        eligible_family_set = set(headline_items_df["family_id"].unique().tolist())

    family_metrics_df = metrics_df[metrics_df["family_id"].isin(list(eligible_family_set))].copy()

    if metrics_df.empty:
        print("[ERROR] No parsable items to score.", file=sys.stderr)
        sys.exit(1)

    metrics = compute_all_metrics(metrics_df, family_metrics_df=family_metrics_df)
    final, eff_w = composite_score_with_effective_weights(metrics)

    # Attach overall columns to the *wide* df (repeat on every row).
    out_df = df.copy()
    out_df["overall_FinalScore"] = final
    out_df["overall_CAA_alpha_0_25"] = metrics.get("core_confidence_accuracy")
    out_df["overall_I_MEFR"] = metrics.get("I_MEFR")
    out_df["overall_T_MEFR"] = metrics.get("T_MEFR")
    out_df["overall_MEFR"] = metrics.get("MEFR")
    out_df["overall_AURC"] = metrics.get("abstention_quality")
    out_df["overall_ToMAcc"] = metrics.get("second_order_accuracy")
    out_df["overall_strict_3cell"] = strict
    out_df["overall_total_done_families"] = int(len(done_df))
    out_df["overall_total_cells_4x_done"] = int(4 * len(done_df))
    out_df["overall_unparsable_items"] = int(unparsable_items)
    out_df["overall_families_with_any_unparsable"] = int(len(families_with_any_unparsable))
    out_df["overall_families_dropped_base_label_nonbinary"] = int(dropped_families_bad_gold)
    out_df["overall_families_in_mefr_denoms"] = int(len(eligible_family_set))
    out_df["overall_items_scored_headline_3cell"] = int(len(metrics_df))
    out_df["overall_effective_w_CAA"] = eff_w.get("core_confidence_accuracy")
    out_df["overall_effective_w_MEFR"] = eff_w.get("MEFR")
    out_df["overall_effective_w_AURC"] = eff_w.get("abstention_quality")
    out_df["overall_effective_w_ToM"] = eff_w.get("second_order_accuracy")

    out_path = args.out
    if out_path is None:
        out_path = f"{args.input}.with_scores.csv"

    out_df.to_csv(out_path, index=False)
    print(f"[INFO] Wrote annotated wide CSV: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()


