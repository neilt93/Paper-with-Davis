"""
Append a single-row score summary to a Google Sheets "scores" tab.

This keeps inference outputs (the 4 XOR JSON columns) separate from scoring,
and makes scores easy to track over time (incremental append).

Usage example:
  python scripts/scoring/append_score_to_sheet.py \
    --spreadsheet-id <ID> \
    --scores-sheet "Scores" \
    --input "Sheets/Pictures DB.gpt.vlm.csv" \
    --model gpt \
    --strict-3cell
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Optional: load GOOGLE_APPLICATION_CREDENTIALS and other env vars from .env
try:  # pragma: no cover
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # type: ignore

if load_dotenv is not None:
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()

# Make it easy to import sibling scoring modules when run as a script.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from metrics import (
    compute_all_metrics,
    composite_score_with_effective_weights,
    aurc_answered_only_diagnostics,
)  # type: ignore
from score_vlm_run import extract_json_fields, _normalise_label, _normalise_confidence  # type: ignore


def get_gspread_client(creds_path: str):
    try:
        import gspread  # type: ignore
        from google.oauth2.service_account import Credentials  # type: ignore
    except ImportError:
        raise RuntimeError("Missing gspread/google-auth. Install with `pip install gspread google-auth`.")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
    client = gspread.authorize(creds)
    return client


def _xor_gold_fixed() -> Dict[str, str]:
    """Fixed XOR gold mapping for this dataset (BASE is always NOT_VISIBLE)."""
    return {"BASE": "NOT_VISIBLE", "TEXT_FLIP": "VISIBLE", "IMAGE_FLIP": "VISIBLE", "DOUBLE_FLIP": "NOT_VISIBLE"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Append one run's score summary to a Google Sheets tab.")
    parser.add_argument("--spreadsheet-id", required=True, help="Google Sheets spreadsheet ID.")
    parser.add_argument("--scores-sheet", default="Scores", help="Tab name to append rows to (default: Scores).")
    parser.add_argument("--creds", default=None, help="Path to service account JSON (default: GOOGLE_APPLICATION_CREDENTIALS).")
    parser.add_argument("--input", "-i", required=True, help="Path to <sheet>.<model>.vlm.csv")
    parser.add_argument("--model", "-m", required=True, help="Model prefix (e.g. gpt).")
    parser.add_argument(
        "--strict-3cell",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled (default), require BASE/TEXT_FLIP/IMAGE_FLIP parsable for families to enter MEFR denominators.",
    )
    args = parser.parse_args()

    creds_path = args.creds or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        print("[ERROR] No Google credentials. Set GOOGLE_APPLICATION_CREDENTIALS or pass --creds.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"[ERROR] Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    if "id" not in df.columns and "ID" in df.columns:
        df["id"] = df["ID"]

    if "id" not in df.columns:
        print("[ERROR] Missing ID/id column.", file=sys.stderr)
        sys.exit(1)

    # Done families only
    done_df = df.copy()
    if "Status" in done_df.columns:
        done_df = done_df[done_df["Status"].astype(str).str.strip() == "Done"]
    if done_df.empty:
        print("[ERROR] No Status==Done rows.", file=sys.stderr)
        sys.exit(1)

    model = args.model
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

    total_families = int(len(done_df))
    total_cells = 4 * total_families

    unparsable_items = 0  # includes empty/NaN, invalid label, answered missing confidence
    dropped_families_bad_gold = 0
    families_with_any_unparsable: set[str] = set()
    complete_headline_families: set[str] = set()
    unparsable_items_by_rel: Dict[str, int] = {rel: 0 for rel in col_map.values()}

    rows_long: list[dict[str, Any]] = []
    for _, row in done_df.iterrows():
        fam = str(row["id"])
        is_tom = 1 if fam.startswith("MA-") else 0

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
            answered = pred_n in {"VISIBLE", "NOT_VISIBLE"}
            missing_conf_answered_cell = answered and (conf_n is None)
            if pred_n is None or is_empty or missing_conf_answered_cell:
                unparsable_items += 1
                unparsable_items_by_rel[rel] += 1
                families_with_any_unparsable.add(fam)

        golds = _xor_gold_fixed()

        if all(parsed_labels.get(rel) is not None for rel in ("BASE", "TEXT_FLIP", "IMAGE_FLIP")):
            complete_headline_families.add(fam)

        for rel in ("BASE", "TEXT_FLIP", "IMAGE_FLIP"):
            pred_n = parsed_labels.get(rel)
            conf_n = parsed_confs.get(rel)
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
    if items_df.empty:
        print("[ERROR] No parsable headline items.", file=sys.stderr)
        sys.exit(1)

    # Confidence required only for answered predictions.
    metrics_df = items_df.copy()
    answered_mask = metrics_df["pred_label"].astype(str).str.upper().isin(["VISIBLE", "NOT_VISIBLE"])
    missing_conf_answered = int((answered_mask & metrics_df["confidence"].isna()).sum())
    metrics_df = metrics_df[~answered_mask | metrics_df["confidence"].notna()]

    strict = bool(args.strict_3cell)
    if strict:
        eligible_family_set = complete_headline_families
    else:
        eligible_family_set = set(items_df["family_id"].unique().tolist())

    family_metrics_df = metrics_df[metrics_df["family_id"].isin(list(eligible_family_set))].copy()

    metrics = compute_all_metrics(metrics_df, family_metrics_df=family_metrics_df)
    final, eff_w = composite_score_with_effective_weights(metrics)
    aurc_diag = aurc_answered_only_diagnostics(
        metrics_df["gold_label"].values,
        metrics_df["pred_label"].values,
        metrics_df["confidence"].values,
    )

    abstain_count = int((metrics_df["pred_label"].astype(str).str.upper() == "ABSTAIN").sum())
    tom_items = int(metrics_df[metrics_df["is_second_order_tom"].astype(bool)].shape[0])

    row_out = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "input_file": args.input,
        "model": model,
        "strict_3cell": strict,
        "FinalScore": final,
        "CAA_alpha_0_25": metrics.get("core_confidence_accuracy"),
        "I_MEFR": metrics.get("I_MEFR"),
        "T_MEFR": metrics.get("T_MEFR"),
        "MEFR": metrics.get("MEFR"),
        "AURC": metrics.get("abstention_quality"),
        "ToMAcc": metrics.get("second_order_accuracy"),
        "total_done_families": total_families,
        "total_cells_4x_done": total_cells,
        "items_scored_headline_3cell": int(len(metrics_df)),
        "abstain_count": abstain_count,
        "unparsable_items": unparsable_items,
        "families_with_any_unparsable": int(len(families_with_any_unparsable)),
        "unparsable_BASE": int(unparsable_items_by_rel["BASE"]),
        "unparsable_TEXT_FLIP": int(unparsable_items_by_rel["TEXT_FLIP"]),
        "unparsable_IMAGE_FLIP": int(unparsable_items_by_rel["IMAGE_FLIP"]),
        "unparsable_DOUBLE_FLIP": int(unparsable_items_by_rel["DOUBLE_FLIP"]),
        "families_dropped_base_label_nonbinary": dropped_families_bad_gold,
        "families_in_mefr_denoms": int(len(eligible_family_set)),
        "dropped_answered_missing_confidence": missing_conf_answered,
        "aurc_n_answered": int(aurc_diag["n_answered"]) if aurc_diag["n_answered"] == aurc_diag["n_answered"] else None,
        "aurc_answered_accuracy_p": aurc_diag["answered_accuracy"],
        "aurc_A_model": aurc_diag["A_model"],
        "aurc_raw_normalized": aurc_diag["raw_normalized"],
        "effective_w_CAA": eff_w.get("core_confidence_accuracy"),
        "effective_w_MEFR": eff_w.get("MEFR"),
        "effective_w_AURC": eff_w.get("abstention_quality"),
        "effective_w_ToM": eff_w.get("second_order_accuracy"),
    }

    # Append to Google Sheets
    client = get_gspread_client(creds_path)
    sh = client.open_by_key(args.spreadsheet_id)
    try:
        ws = sh.worksheet(args.scores_sheet)
    except Exception:
        ws = sh.add_worksheet(title=args.scores_sheet, rows=1, cols=1)

    existing = ws.get_all_values()
    header = list(row_out.keys())

    if not existing:
        ws.resize(rows=1, cols=len(header))
        ws.update(range_name="A1", values=[header])
    else:
        # If header differs, overwrite header row to keep consistent schema.
        if existing and existing[0] != header:
            ws.resize(rows=max(ws.row_count, 1), cols=len(header))
            ws.update(range_name="A1", values=[header])

    # Convert NaN/None to None (which becomes empty cell in Sheets) for JSON serialization
    import math
    row_values = []
    for k in header:
        val = row_out[k]
        if val is not None and isinstance(val, float) and math.isnan(val):
            val = None
        row_values.append(val)
    ws.append_row(row_values, value_input_option="RAW")
    print(f"[INFO] Appended score row to sheet '{args.scores_sheet}'", file=sys.stderr)


if __name__ == "__main__":
    main()


