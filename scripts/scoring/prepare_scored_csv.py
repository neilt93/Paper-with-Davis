"""Prepare a *.vlm.csv file for scoring by extracting gold/pred/confidence.

This script is the glue between the raw model outputs (JSON blobs in
<model>_base_json / <model>_flip_json) and the scoring script, which expects
flat columns:

  - gold_label
  - pred_label
  - confidence

Usage example for LLaVA-OneVision on base images:

    python scripts/scoring/prepare_scored_csv.py \
        --input "Sheets/Pictures DB.llava-onevision.vlm.csv" \
        --model llava-onevision \
        --side base

This will:
  - use `base_label` as the gold label column for side=base
  - parse `llava-onevision_base_json` into:
        llava-onevision_base_label
        llava-onevision_base_confidence
        llava-onevision_base_reason_code
  - also populate generic:
        gold_label
        pred_label
        confidence
    so that `score_dataset.py` can run with its defaults.

For side=flip, it will use `flip_label` and `<model>_flip_json` instead.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd


def extract_json_fields(raw: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """Best-effort parse of the model's JSON string.

    Returns (label, confidence, reason_code). If parsing fails, all fields
    are returned as None.
    """
    if not isinstance(raw, str):
        return None, None, None

    text = raw.strip()
    if not text:
        return None, None, None

    # Try to locate the first {...} block to be robust to extra text.
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        candidate = text[start:end]
    except ValueError:
        # No braces found
        candidate = text

    try:
        obj = json.loads(candidate)
    except Exception:
        return None, None, None

    label = obj.get("label")
    conf = obj.get("confidence")
    reason = obj.get("reason_code")

    try:
        if conf is not None:
            conf = float(conf)
    except Exception:
        conf = None

    return label, conf, reason


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a VLM output CSV for scoring by adding gold_label/pred_label/confidence."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the *.vlm.csv file for a given model.",
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name prefix used in column names (e.g. 'llava-onevision', 'qwen-vl').",
    )
    parser.add_argument(
        "--side",
        choices=["base", "flip"],
        default="base",
        help="Which side to score: base or flip (default: base).",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Output CSV path (default: <input_basename>.scored.csv).",
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

    model = args.model
    side = args.side

    # Decide which gold and JSON columns to use.
    if side == "base":
        gold_col = "base_label"
        json_col = f"{model}_base_json"
        pred_label_col = f"{model}_base_label"
        conf_col = f"{model}_base_confidence"
        reason_col = f"{model}_base_reason_code"
    else:
        gold_col = "flip_label"
        json_col = f"{model}_flip_json"
        pred_label_col = f"{model}_flip_label"
        conf_col = f"{model}_flip_confidence"
        reason_col = f"{model}_flip_reason_code"

    if gold_col not in df.columns:
        print(f"[ERROR] Expected gold label column '{gold_col}' not found in CSV.", file=sys.stderr)
        sys.exit(1)

    if json_col not in df.columns:
        print(f"[ERROR] Expected JSON output column '{json_col}' not found in CSV.", file=sys.stderr)
        sys.exit(1)

    # Parse JSON outputs row-by-row.
    labels = []
    confidences = []
    reasons = []

    for raw in df[json_col].tolist():
        label, conf, reason = extract_json_fields(raw)
        labels.append(label)
        confidences.append(conf)
        reasons.append(reason)

    df[pred_label_col] = labels
    df[conf_col] = confidences
    df[reason_col] = reasons

    # Also populate the generic columns expected by score_dataset.py.
    df["gold_label"] = df[gold_col]
    df["pred_label"] = df[pred_label_col]
    df["confidence"] = df[conf_col]

    out_path = args.out
    if out_path is None:
        base, ext = os.path.splitext(args.input)
        out_path = f"{base}.scored.csv"

    try:
        df.to_csv(out_path, index=False)
    except Exception as e:
        print(f"[ERROR] Failed to write scored CSV '{out_path}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Wrote scored CSV to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()


