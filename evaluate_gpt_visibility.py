import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd


@dataclass
class Sample:
    """One image-level sample (base or flip)."""
    row_id: str        # e.g. AV-01
    which: str         # "base" or "flip"
    gold_label: str    # VISIBLE / NOT_VISIBLE / UNDETERMINABLE_FROM_EVIDENCE
    pred_label: Optional[str]
    confidence: Optional[float]
    category: str      # for ToM filtering


def parse_pred_cell(cell: str) -> Tuple[Optional[str], Optional[float]]:
    """
    cell is something like:
      {"label":"NOT_VISIBLE","reason_code":"OUT_OF_FRAME","confidence":0.88}
    Return (label, confidence). On failure, (None, None).
    """
    if not isinstance(cell, str) or not cell.strip():
        return None, None
    try:
        obj = json.loads(cell)
        label = obj.get("label")
        conf = obj.get("confidence")
        if conf is not None:
            conf = float(conf)
        return label, conf
    except Exception:
        return None, None


def build_samples(df_done: pd.DataFrame) -> List[Sample]:
    samples: List[Sample] = []
    for _, row in df_done.iterrows():
        row_id = str(row["id"])
        category = str(row.get("category", ""))

        # Base image sample
        base_gold = row["base_label"]
        base_pred_label, base_conf = parse_pred_cell(row.get("gpt_output_base", ""))
        samples.append(
            Sample(
                row_id=row_id,
                which="base",
                gold_label=str(base_gold),
                pred_label=base_pred_label,
                confidence=base_conf,
                category=category,
            )
        )

        # Flip image sample
        flip_gold = row["flip_label"]
        flip_pred_label, flip_conf = parse_pred_cell(row.get("gpt_output_flip", ""))
        samples.append(
            Sample(
                row_id=row_id,
                which="flip",
                gold_label=str(flip_gold),
                pred_label=flip_pred_label,
                confidence=flip_conf,
                category=category,
            )
        )
    return samples


# ---------- A: Confidence-aware accuracy ----------

def confidence_aware_accuracy(samples: List[Sample]) -> Tuple[float, List[Tuple[str, str, float]]]:
    """
    Score per sample:
      correct: score = conf
      wrong:   score = 1 - conf
    Return (mean_score, per_sample_scores_for_debug)
    """
    per_scores = []
    usable_scores = []

    for s in samples:
        if s.pred_label is None or s.confidence is None:
            continue
        correct = (s.pred_label == s.gold_label)
        score_i = s.confidence if correct else 1.0 - s.confidence
        usable_scores.append(score_i)
        per_scores.append((s.row_id, s.which, score_i))

    if not usable_scores:
        return 0.0, per_scores

    return sum(usable_scores) / len(usable_scores), per_scores


# ---------- C: Minimal-Edit Flip Rate (MEFR) ----------

def mefr(df_done: pd.DataFrame) -> float:
    """
    For each row, we have a base and flip example.
    We count a flip as correct (1) if:
      - base prediction label matches base gold label, AND
      - flip prediction label matches flip gold label.
    """
    scores = []

    for _, row in df_done.iterrows():
        base_gold = str(row["base_label"])
        flip_gold = str(row["flip_label"])

        base_pred, _ = parse_pred_cell(row.get("gpt_output_base", ""))
        flip_pred, _ = parse_pred_cell(row.get("gpt_output_flip", ""))

        if base_pred is None or flip_pred is None:
            continue  # skip if missing

        correct_flip = (base_pred == base_gold) and (flip_pred == flip_gold)
        scores.append(1.0 if correct_flip else 0.0)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


# ---------- D: Abstention quality (normalized AURC) ----------

def normalized_aurc(samples: List[Sample]) -> float:
    """
    Compute area under risk-coverage curve (AURC), then normalize:
      0 = random baseline, 1 = oracle.
    Confidence is used as ranking; we don't explicitly abstain but simulate thresholds.
    """
    items = []
    for s in samples:
        if s.pred_label is None or s.confidence is None:
            continue
        correct = (s.pred_label == s.gold_label)
        items.append((s.confidence, correct))

    if not items:
        return 0.0

    # Sort by descending confidence
    items.sort(key=lambda x: x[0], reverse=True)
    N = len(items)

    total_correct = sum(1 for _, c in items if c)
    error_rate = 1.0 - total_correct / N

    # Raw AURC
    cum_correct = 0
    risks = []
    for k, (_, correct) in enumerate(items, start=1):
        if correct:
            cum_correct += 1
        risk_k = 1.0 - (cum_correct / k)
        risks.append(risk_k)
    raw_aurc = sum(risks) / N

    # Oracle AURC: all correct first
    C = total_correct
    if C == 0:
        return 0.0
    if C == N:
        return 1.0

    oracle_risks = []
    for k in range(1, N + 1):
        if k <= C:
            oracle_risks.append(0.0)
        else:
            oracle_risks.append(1.0 - C / k)
    oracle_aurc = sum(oracle_risks) / N

    # Random baseline: constant risk = error_rate
    random_aurc = error_rate

    denom = random_aurc - oracle_aurc
    if abs(denom) < 1e-12:
        return 0.0

    norm_score = (random_aurc - raw_aurc) / denom
    return max(0.0, min(1.0, norm_score))


# ---------- E: Second-order ToM accuracy ----------

def tom_accuracy(samples: List[Sample]) -> Optional[float]:
    """
    Filter samples whose category indicates second-order ToM,
    e.g. contains 'MULTI_AGENT' or 'SECOND_ORDER' (case-insensitive).
    """
    tom_samples = []
    for s in samples:
        cat_up = s.category.upper()
        if "MULTI_AGENT" in cat_up or "SECOND_ORDER" in cat_up:
            if s.pred_label is None:
                continue
            tom_samples.append(s)

    if not tom_samples:
        return None

    correct_flags = [(s.pred_label == s.gold_label) for s in tom_samples]
    return sum(correct_flags) / len(correct_flags)


# ---------- Final composite score ----------

def final_composite_score(
    conf_acc: float,
    mefr_score: float,
    aurc_score: float,
    tom_acc: Optional[float],
) -> float:
    """
    Final Score =
      70% × Confidence-Aware Accuracy
      15% × Minimal-Edit Flip Rate
      10% × Abstention Quality
       5% × Second-order ToM

    If some metrics are unavailable (e.g. no ToM items),
    renormalize weights over available ones.
    """
    weights = {
        "conf": 0.70,
        "mefr": 0.15,
        "aurc": 0.10,
        "tom": 0.05,
    }
    scores = {
        "conf": conf_acc,
        "mefr": mefr_score,
        "aurc": aurc_score,
        "tom": tom_acc,
    }

    available_keys = [k for k, v in scores.items() if v is not None]
    total_weight = sum(weights[k] for k in available_keys)
    if total_weight == 0.0:
        return 0.0

    weighted_sum = sum(weights[k] * scores[k] for k in available_keys)
    return weighted_sum / total_weight


# ---------- Main script ----------

def main():
    parser = argparse.ArgumentParser(
        description="Compute GPT visibility benchmark scores from CSV and print markdown tables."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="11-13-25.csv",
        help="Input CSV file (default: 11-13-25.csv)",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Optional output CSV file with per-row gpt_score",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)

    required_cols = {
        "id",
        "Status",
        "base_label",
        "flip_label",
        "gpt_output_base",
        "gpt_output_flip",
        "category",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Missing required columns: {', '.join(sorted(missing))}", file=sys.stderr)
        sys.exit(1)

    # Only use rows marked Done
    df_done = df[df["Status"] == "Done"].copy()
    if df_done.empty:
        print("No rows with Status == 'Done'.", file=sys.stderr)
        sys.exit(1)

    # Build image-level samples
    samples = build_samples(df_done)

    # A: Confidence-aware accuracy
    conf_acc, per_sample_scores = confidence_aware_accuracy(samples)

    # C: MEFR
    mefr_score = mefr(df_done)

    # D: Abstention quality
    aurc_score = normalized_aurc(samples)

    # E: ToM accuracy
    tom_acc = tom_accuracy(samples)

    # Final composite
    final_score = final_composite_score(conf_acc, mefr_score, aurc_score, tom_acc)

    # ------- Per-row gpt_score (avg of base+flip confidence-aware scores) -------
    per_sample_map = {(rid, which): score for rid, which, score in per_sample_scores}

    row_scores = []
    for _, row in df_done.iterrows():
        row_id = str(row["id"])
        sb = per_sample_map.get((row_id, "base"))
        sf = per_sample_map.get((row_id, "flip"))

        if sb is not None and sf is not None:
            row_score = 0.5 * (sb + sf)
        elif sb is not None:
            row_score = sb
        elif sf is not None:
            row_score = sf
        else:
            row_score = None
        row_scores.append((row_id, row_score))

    # Attach to df_done if we want to save
    df_done = df_done.copy()
    row_score_map = {rid: sc for rid, sc in row_scores if sc is not None}
    df_done["gpt_score"] = df_done["id"].map(row_score_map)

    if args.out:
        df_done.to_csv(args.out, index=False)

    # ----------------- PRINT MARKDOWN -----------------

    # Overall metrics
    print("\n### Overall GPT Metrics\n")
    print("| Metric | Value |")
    print("| --- | --- |")
    print(f"| Confidence-aware accuracy (A) | {conf_acc:.4f} |")
    print(f"| Minimal-Edit Flip Rate (MEFR, C) | {mefr_score:.4f} |")
    print(f"| Abstention quality (normalized AURC, D) | {aurc_score:.4f} |")
    if tom_acc is not None:
        print(f"| Second-order ToM accuracy (E) | {tom_acc:.4f} |")
    else:
        print("| Second-order ToM accuracy (E) | N/A |")
    print(f"| **Final composite score** | **{final_score:.4f}** |")

    # Per-row table
    print("\n### Per-row GPT Scores\n")
    print("| id | gpt_score |")
    print("| --- | --- |")
    for rid, sc in row_scores:
        if sc is None:
            print(f"| {rid} |  |")
        else:
            print(f"| {rid} | {sc:.4f} |")


if __name__ == "__main__":
    main()


