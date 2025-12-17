"""One-shot scoring for a single VLM run on the visibility dataset.

Given a `<model>.vlm.csv` file produced by `run_visibility_eval.py`, this
script will:
  1) Parse the model's JSON outputs (<model>_base_json / _flip_json) into
     label / confidence / reason_code columns.
  2) Populate generic `gold_label`, `pred_label`, `confidence` columns.
  3) Compute all scalar metrics and the composite score.
  4) Save a `<input>.scored.csv` file with the added columns.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# Ensure we can import metrics when run as a standalone script
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from metrics import compute_all_metrics, composite_score_with_effective_weights, aurc_answered_only_diagnostics  # type: ignore


VALID_LABELS = {
    "VISIBLE",
    "NOT_VISIBLE",
    "ABSTAIN",
    "UNDETERMINABLE_FROM_EVIDENCE",  # backward compat
}


def _normalise_label(x: Any) -> Optional[str]:
    if x is None:
        return None
    if not isinstance(x, str):
        return None
    s = x.strip().upper()
    # Common variants
    if s in {"UNDETERMINABLE", "UNDETERMINABLE_FROM_THE_EVIDENCE", "UNDETERMINABLE_FROM_EVIDENCE"}:
        s = "ABSTAIN"
    if s in VALID_LABELS:
        return s
    return s if s else None


def _normalise_confidence(x: Any) -> Optional[float]:
    if x is None:
        return None
    # Strings like "0.73", "73", "73%", "0.73\n"
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        s = s.replace("%", "").strip()
        try:
            v = float(s)
        except Exception:
            return None
    else:
        try:
            v = float(x)
        except Exception:
            return None

    # If model outputs 73 instead of 0.73, interpret as percent
    if v > 1.0 and v <= 100.0:
        v = v / 100.0

    # Clip to [0, 1]
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0

    return v


def _strip_code_fences(text: str) -> str:
    # Removes ```json ... ``` wrappers if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _find_first_json_object(text: str) -> Optional[str]:
    """Best-effort extraction of the first {...} JSON object in a string."""
    if not text:
        return None

    # Quick path: starts with { and ends with }
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return t

    # Otherwise scan for a balanced object
    start = t.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return t[start : i + 1]
    return None


def _maybe_unwrap(obj: Any) -> Any:
    """Some runners wrap outputs, eg {response:{...}} or {output:{...}}."""
    if not isinstance(obj, dict):
        return obj
    for k in ("response", "output", "result", "answer"):
        v = obj.get(k)
        if isinstance(v, dict) and ("label" in v or "confidence" in v or "reason_code" in v):
            return v
    return obj


def extract_json_fields(raw: Any) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """Parse model JSON string. Returns (label, confidence, reason_code)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None, None, None
    if not isinstance(raw, str):
        raw = str(raw)

    text = _strip_code_fences(raw)
    if not text:
        return None, None, None

    candidate = _find_first_json_object(text)
    if candidate is None:
        # As a fallback try to parse the entire thing
        candidate = text

    try:
        obj = json.loads(candidate)
    except Exception:
        return None, None, None

    obj = _maybe_unwrap(obj)

    if not isinstance(obj, dict):
        return None, None, None

    label = _normalise_label(obj.get("label"))
    conf = _normalise_confidence(obj.get("confidence"))
    reason = obj.get("reason_code")
    reason = reason.strip() if isinstance(reason, str) else (str(reason) if reason is not None else None)

    return label, conf, reason


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse a <model>.vlm.csv file, add scoring columns, compute metrics, "
            "and write <input>.scored.csv."
        )
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the *.vlm.csv file.")
    parser.add_argument("--model", "-m", required=True, help="Model name prefix used in column names.")
    parser.add_argument(
        "--side",
        choices=["xor"],
        default="xor",
        help=(
            "Scoring mode (XOR only): expects 2x2 columns "
            "<model>_I0q0_json/_I0q1_json/_I1q0_json/_I1q1_json and scores the XOR family metrics."
        ),
    )
    parser.add_argument("--family-col", default="family_id", help="Family/group column.")
    parser.add_argument("--variant-col", default="variant_relation", help="Variant relation column.")
    parser.add_argument("--tom-flag-col", default="is_second_order_tom", help="Second-order ToM flag column.")
    parser.add_argument(
        "--strict-3cell",
        dest="strict_3cell",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If enabled (default), require the 3 headline XOR cells (I0q0/I0q1/I1q0 i.e. "
            "BASE/TEXT_FLIP/IMAGE_FLIP) to be parsable for a family to be included in MEFR denominators. "
            "CAA/AURC always just drop unparsable items. DOUBLE_FLIP (I1q1) is diagnostic-only."
        ),
    )
    # Backwards-compatible alias (deprecated)
    parser.add_argument(
        "--strict-2x2",
        dest="strict_3cell",
        action=argparse.BooleanOptionalAction,
        help="(Deprecated alias for --strict-3cell) Use --strict-3cell / --no-strict-3cell.",
    )
    parser.add_argument("--out", "-o", default=None, help="Output CSV path.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input CSV not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV '{args.input}': {e}", file=sys.stderr)
        sys.exit(1)

    model = args.model
    side = args.side

    # XOR gold is fixed in this dataset (BASE is always NOT_VISIBLE), so `base_label`
    # is optional. If present and binary, we still support it; otherwise we default
    # to NOT_VISIBLE.

    # XOR mode: explode 4 cells per family
    if "id" not in df.columns and "ID" in df.columns:
        df["id"] = df["ID"]
    if "id" not in df.columns:
        print("[ERROR] Expected an 'id' or 'ID' column for family ids.", file=sys.stderr)
        sys.exit(1)

    col_map = {
        f"{model}_I0q0_json": "BASE",
        f"{model}_I0q1_json": "TEXT_FLIP",
        f"{model}_I1q0_json": "IMAGE_FLIP",
        f"{model}_I1q1_json": "DOUBLE_FLIP",
    }
    missing_json = [c for c in col_map.keys() if c not in df.columns]
    if missing_json:
        print(
            "[ERROR] Missing XOR 2x2 JSON columns: " + ", ".join(missing_json),
            file=sys.stderr,
        )
        sys.exit(1)

    def xor_gold_from_base(base_label: Any) -> Dict[str, str]:
        """
        XOR gold mapping.

        Default assumption: BASE is always NOT_VISIBLE (so flips are VISIBLE and double-flip returns to NOT_VISIBLE).
        If a binary `base_label` is present (VISIBLE/NOT_VISIBLE), we respect it.
        """
        b = _normalise_label(base_label)
        if b == "VISIBLE":
            return {"BASE": "VISIBLE", "TEXT_FLIP": "NOT_VISIBLE", "IMAGE_FLIP": "NOT_VISIBLE", "DOUBLE_FLIP": "VISIBLE"}
        # Default / fallback: NOT_VISIBLE
        return {"BASE": "NOT_VISIBLE", "TEXT_FLIP": "VISIBLE", "IMAGE_FLIP": "VISIBLE", "DOUBLE_FLIP": "NOT_VISIBLE"}

    # Restrict to Done families for scoring.
    done_df = df.copy()
    if "Status" in done_df.columns:
        done_df = done_df[done_df["Status"].astype(str).str.strip() == "Done"]

    total_families = int(len(done_df))
    total_cells = 4 * total_families

    # Track parse robustness across *all* Done families.
    families_with_any_unparsable: set[str] = set()
    unparsable_items = 0  # includes empty/NaN, JSON parse failure, invalid label, missing conf for answered
    unparsable_items_by_rel: Dict[str, int] = {rel: 0 for rel in col_map.values()}
    empty_items_by_rel: Dict[str, int] = {rel: 0 for rel in col_map.values()}
    missing_conf_answered_by_rel: Dict[str, int] = {rel: 0 for rel in col_map.values()}

    # Build the item table for all Done families (gold is fixed / well-defined).
    rows_long: list[dict[str, Any]] = []
    dropped_items_unparsed_scored = 0  # unparsable cells among families with XOR gold

    # For strict family metrics (MEFR), track which families have all 3 headline cells parsable.
    complete_headline_families: set[str] = set()
    # For DFAcc diagnostic, track which families have BASE+DOUBLE_FLIP parsable.
    complete_dfacc_families: set[str] = set()

    for _, row in done_df.iterrows():
        fam = str(row["id"])
        is_tom = 1 if re.match(r"^MA-", fam) else 0

        # Parse all 4 cells to compute robustness counts.
        parsed_labels: Dict[str, Optional[str]] = {}
        parsed_confs: Dict[str, Optional[float]] = {}
        for col_name, rel in col_map.items():
            raw_cell = row.get(col_name)
            is_empty = raw_cell is None or (isinstance(raw_cell, float) and pd.isna(raw_cell)) or (isinstance(raw_cell, str) and not raw_cell.strip())

            pred, conf, _reason = extract_json_fields(raw_cell)
            pred_n = _normalise_label(pred)
            conf_n = _normalise_confidence(conf)
            parsed_labels[rel] = pred_n
            parsed_confs[rel] = conf_n
            # Count as "unparsable" if empty OR missing/invalid label OR answered missing confidence.
            answered = pred_n in {"VISIBLE", "NOT_VISIBLE"}
            missing_conf_answered = answered and (conf_n is None)

            if is_empty:
                empty_items_by_rel[rel] += 1
            if pred_n is None or is_empty or missing_conf_answered:
                unparsable_items += 1
                families_with_any_unparsable.add(fam)
                unparsable_items_by_rel[rel] += 1
            if missing_conf_answered:
                missing_conf_answered_by_rel[rel] += 1

        golds = xor_gold_from_base(row.get("base_label") if "base_label" in row else None)

        # Family completeness is defined only for families that are eligible for XOR gold.
        if all(parsed_labels[rel] is not None for rel in ("BASE", "TEXT_FLIP", "IMAGE_FLIP")):
            complete_headline_families.add(fam)
        if (parsed_labels.get("BASE") is not None) and (parsed_labels.get("DOUBLE_FLIP") is not None):
            complete_dfacc_families.add(fam)

        for rel in col_map.values():
            pred_n = parsed_labels[rel]
            conf_n = parsed_confs[rel]
            if pred_n is None:
                dropped_items_unparsed_scored += 1
                continue
            rows_long.append(
                {
                    "family_id": fam,
                    "is_second_order_tom": is_tom,
                    "variant_relation": rel,
                    "gold_label": golds[rel],
                    "pred_label": pred_n,
                    "confidence": conf_n,
                    "Status": "Done",
                }
            )

    items_df = pd.DataFrame(rows_long)

    # Headline metrics use only 3 cells: BASE/TEXT_FLIP/IMAGE_FLIP (DOUBLE_FLIP is diagnostic only).
    headline_items_df = items_df[items_df["variant_relation"].isin(["BASE", "TEXT_FLIP", "IMAGE_FLIP"])].copy()

    # Base pool for CAA/AURC/ToM: all parsable headline items for XOR-eligible families.
    metrics_df = headline_items_df.copy()

    # Confidence is only required for answered predictions; abstains can be scored with alpha.
    metrics_df = metrics_df[metrics_df["gold_label"].notna() & metrics_df["pred_label"].notna()]
    answered_mask = metrics_df["pred_label"].astype(str).str.upper().isin(["VISIBLE", "NOT_VISIBLE"])
    missing_conf_answered = int((answered_mask & metrics_df["confidence"].isna()).sum())
    metrics_df = metrics_df[~answered_mask | metrics_df["confidence"].notna()]

    # For family flip metrics, optionally enforce strict completeness over the 3 headline cells.
    strict = bool(args.strict_3cell)
    if strict:
        eligible_family_set = complete_headline_families
    else:
        eligible_family_set = set(headline_items_df["family_id"].unique().tolist())

    family_metrics_df = metrics_df[metrics_df["family_id"].isin(list(eligible_family_set))].copy()

    families_eligible_for_xor = total_families
    families_excluded_from_mefr = int(families_eligible_for_xor - len(eligible_family_set)) if strict else 0

    if metrics_df.empty:
        print("[ERROR] No parsable items to score.", file=sys.stderr)
        sys.exit(1)

    # DFAcc diagnostic: computed on BASE + DOUBLE_FLIP, requires both cells to be parsable.
    # Not part of headline metrics nor FinalScore.
    dfacc = float("nan")
    dfacc_den = 0
    dfacc_num = 0
    if complete_dfacc_families:
        dfacc_df = items_df[
            items_df["family_id"].isin(list(complete_dfacc_families))
            & items_df["variant_relation"].isin(["BASE", "DOUBLE_FLIP"])
        ].copy()
        if not dfacc_df.empty:
            base_ok = (
                dfacc_df[dfacc_df["variant_relation"] == "BASE"]
                .groupby("family_id")
                .apply(lambda g: bool((g["gold_label"] == g["pred_label"]).all()))
            )
            double_ok = (
                dfacc_df[dfacc_df["variant_relation"] == "DOUBLE_FLIP"]
                .groupby("family_id")
                .apply(lambda g: bool((g["gold_label"] == g["pred_label"]).all()))
            )
            eligible = base_ok[base_ok].index.intersection(double_ok.index)
            dfacc_den = int(len(eligible))
            if dfacc_den > 0:
                dfacc_num = int(double_ok.loc[eligible].sum())
                dfacc = float(dfacc_num / dfacc_den)

    try:
        metrics = compute_all_metrics(
            metrics_df,
            gold_col="gold_label",
            pred_col="pred_label",
            confidence_col="confidence",
            family_col=args.family_col,
            relation_col=args.variant_col,
            tom_flag_col=args.tom_flag_col,
            family_metrics_df=family_metrics_df,
        )
        final, eff_w = composite_score_with_effective_weights(metrics)
    except Exception as e:
        print(f"[ERROR] Failed to compute metrics: {e}", file=sys.stderr)
        sys.exit(1)

    # No legacy per-row score column in XOR-only mode.

    out_path = args.out
    if out_path is None:
        base, _ext = os.path.splitext(args.input)
        out_path = f"{base}.scored.csv"

    try:
        df.to_csv(out_path, index=False)
    except Exception as e:
        print(f"[ERROR] Failed to write scored CSV '{out_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # Report
    print("=== Visibility Evaluation Scores ===")
    print(f"Input file            : {args.input}")
    print(f"Scored CSV written to : {out_path}")
    print(f"Side                  : {side}")
    print(f"Strict 3-cell (MEFR requires I0q0/I0q1/I1q0) : {strict}")
    print()
    print("Sanity counts:")
    print(f"  Total families (Status==Done)              : {total_families}")
    print(f"  Total cells (4xDone families)              : {total_cells}")
    print(f"  Unparsable items (all Done families)       : {unparsable_items}")
    print(
        "  Unparsable items by cell                  : "
        f"BASE={unparsable_items_by_rel['BASE']} "
        f"TEXT_FLIP={unparsable_items_by_rel['TEXT_FLIP']} "
        f"IMAGE_FLIP={unparsable_items_by_rel['IMAGE_FLIP']} "
        f"DOUBLE_FLIP={unparsable_items_by_rel['DOUBLE_FLIP']}"
    )
    print(
        "  Empty/NaN outputs by cell                 : "
        f"BASE={empty_items_by_rel['BASE']} "
        f"TEXT_FLIP={empty_items_by_rel['TEXT_FLIP']} "
        f"IMAGE_FLIP={empty_items_by_rel['IMAGE_FLIP']} "
        f"DOUBLE_FLIP={empty_items_by_rel['DOUBLE_FLIP']}"
    )
    print(
        "  Answered-missing-confidence by cell       : "
        f"BASE={missing_conf_answered_by_rel['BASE']} "
        f"TEXT_FLIP={missing_conf_answered_by_rel['TEXT_FLIP']} "
        f"IMAGE_FLIP={missing_conf_answered_by_rel['IMAGE_FLIP']} "
        f"DOUBLE_FLIP={missing_conf_answered_by_rel['DOUBLE_FLIP']}"
    )
    print(f"  Families with any unparsable cell          : {len(families_with_any_unparsable)}")
    print("  Families dropped (base_label non-binary)   : 0")
    if strict:
        print(f"  Families excluded from MEFR (missing cells): {families_excluded_from_mefr}")
    print(f"  Families eligible for XOR (binary base)    : {families_eligible_for_xor}")
    print(f"  Families included in MEFR denominators     : {len(eligible_family_set)}")
    print(f"  Items included for CAA/AURC/ToM (3 cells)  : {len(metrics_df)}")
    abstain_count = int((metrics_df["pred_label"].astype(str).str.upper() == "ABSTAIN").sum())
    print(f"  Abstain predictions (ABSTAIN)             : {abstain_count}")
    print(f"  Dropped items (unparsable, XOR-eligible)   : {dropped_items_unparsed_scored}")
    print(f"  Dropped answered items (missing confidence): {missing_conf_answered}")
    tom_items = int(metrics_df[metrics_df[args.tom_flag_col].astype(bool)].shape[0]) if args.tom_flag_col in metrics_df.columns else 0
    print(f"  ToM items used for ToMAcc                  : {tom_items}")
    print()
    print("Component metrics:")
    print(f"  Core confidence-aware accuracy (CAA, alpha=0.25): {metrics['core_confidence_accuracy']:.4f}")
    print(f"  Image flip MEFR (I_MEFR | BASE-correct)    : {metrics['I_MEFR']:.4f}")
    print(f"  Text flip MEFR  (T_MEFR | BASE-correct)    : {metrics['T_MEFR']:.4f}")
    print(f"  Mean conditional flip rate (MEFR)          : {metrics['MEFR']:.4f}")
    print(f"  Double flip accuracy (DFAcc | BASE-correct): {dfacc:.4f}")
    if dfacc_den > 0:
        print(f"  DFAcc denominator (families)              : {dfacc_den} (num={dfacc_num})")
    print(f"  Abstention quality (AURC-normalised)       : {metrics['abstention_quality']:.4f}")
    aurc_diag = aurc_answered_only_diagnostics(
        metrics_df["gold_label"].values,
        metrics_df["pred_label"].values,
        metrics_df["confidence"].values,
    )
    print(
        "  AURC diagnostics (answered-only)          : "
        f"n={int(aurc_diag['n_answered'])} "
        f"p={aurc_diag['answered_accuracy']:.4f} "
        f"A_model={aurc_diag['A_model']:.4f} "
        f"raw={aurc_diag['raw_normalized']:.4f}"
    )
    print(f"  Second-order ToM accuracy                  : {metrics['second_order_accuracy']:.4f}")
    print(f"  Strict invariance consistency (IC_strict)  : {metrics['ic_strict']:.4f}")
    print()
    print("Effective composite weights used (after NaN renormalization):")
    print(f"  w_CAA   : {eff_w['core_confidence_accuracy']:.3f}")
    print(f"  w_MEFR  : {eff_w['MEFR']:.3f}")
    print(f"  w_AURC  : {eff_w['abstention_quality']:.3f}")
    print(f"  w_ToM   : {eff_w['second_order_accuracy']:.3f}")
    print()
    print(f"Final composite score : {final:.4f}")


if __name__ == "__main__":
    main()
