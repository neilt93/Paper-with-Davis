"""
Metric implementations for the visibility evaluation protocol.

This module does NOT depend on any particular model; it only consumes
tabular data (typically a pandas DataFrame) with the following logical
fields:

Required for all metrics:
  - gold_label      : str in {VISIBLE, NOT_VISIBLE, UNDETERMINABLE_FROM_EVIDENCE}
  - pred_label      : same label space as gold_label
  - confidence      : float in [0, 1]

Required for family-structured flip / invariance metrics:
  - family_id       : arbitrary identifier grouping a base and its variants
  - variant_relation: str in {BASE, TEXT_FLIP, IMAGE_FLIP, DOUBLE_FLIP, INVARIANT}

Required for second-order ToM:
  - is_second_order_tom : boolean or 0/1 indicating ToM subset

You can override all column names via the function arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Optional

import numpy as np
import pandas as pd


VALID_LABELS = {"VISIBLE", "NOT_VISIBLE", "ABSTAIN"}


@dataclass
class MetricInputs:
    """Helper dataclass if you prefer to pass structured arrays instead of a DataFrame."""

    gold: Iterable[str]
    pred: Iterable[str]
    confidence: Iterable[float]


def _ensure_numpy(a: Iterable[Any]) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a
    return np.asarray(list(a))


def confidence_aware_accuracy(
    gold: Iterable[str],
    pred: Iterable[str],
    confidence: Iterable[float],
) -> float:
    """
    Core label metric (A): confidence-aware accuracy.

    Per item:
      - if prediction is correct: score_i = confidence_i
      - if prediction is wrong:   score_i = 0

    This avoids gaming where low-confidence wrong answers score well.
    Returns the mean score in [0, 1].
    """
    g = _ensure_numpy(gold)
    p = _ensure_numpy(pred)
    c = _ensure_numpy(confidence).astype(float)

    if g.shape != p.shape or g.shape != c.shape:
        raise ValueError("gold, pred, and confidence must have the same shape.")

    correct = (g == p)
    scores = np.where(correct, c, 0.0)
    return float(scores.mean()) if scores.size else float("nan")


def caa_with_abstain_penalty(
    gold: Iterable[str],
    pred: Iterable[Optional[str]],
    confidence: Iterable[Optional[float]],
    *,
    alpha: float = 0.25,
) -> float:
    """
    Binary CAA with abstention penalty (XOR-style).

    We treat:
      - gold labels: "VISIBLE" or "NOT_VISIBLE"
      - predictions:
          - "VISIBLE"/"NOT_VISIBLE" => answered
          - "ABSTAIN" (or any UNDETERMINABLE* legacy label, or None) => abstain

    Per item i:
      - if abstain: s_i = alpha
      - else if correct: s_i = c_i
      - else: s_i = 0  (no reward for wrong answers)

    Returns mean s_i.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1].")

    g = _ensure_numpy(gold)
    p = _ensure_numpy(pred)
    c = _ensure_numpy(confidence).astype(float)

    if g.shape != p.shape or g.shape != c.shape:
        raise ValueError("gold, pred, and confidence must have the same shape.")

    # Abstain if pred label indicates abstain / refusal.
    p_str = p.astype(object)
    abstain = np.zeros(len(p_str), dtype=bool)
    for i, v in enumerate(p_str):
        if v is None:
            abstain[i] = True
        else:
            s = str(v).strip().upper()
            if s in {"ABSTAIN", "UNDETERMINABLE_FROM_EVIDENCE", "UNDETERMINABLE_FROM_THE_EVIDENCE", "UNDETERMINABLE"}:
                abstain[i] = True

    correct = (g == p)
    scores = np.where(abstain, alpha, np.where(correct, c, 0.0))
    return float(scores.mean()) if scores.size else float("nan")


def _family_level_correct(
    df: pd.DataFrame,
    *,
    gold_col: str,
    pred_col: str,
    family_col: str,
    relation_col: str,
    relation_value: str,
) -> pd.Series:
    """
    Helper: for each family, determine whether *all* items with the given
    `relation_value` are correctly classified.

    Returns a boolean Series indexed by family_id for families that contain
    at least one item with that relation. Families without such items are
    omitted from the result.
    """
    if family_col not in df.columns or relation_col not in df.columns:
        # No family / relation metadata yet.
        return pd.Series(dtype=bool)

    sub = df[df[relation_col] == relation_value]
    if sub.empty:
        return pd.Series(dtype=bool)

    if gold_col not in sub.columns or pred_col not in sub.columns:
        raise ValueError(
            f"gold_col '{gold_col}' or pred_col '{pred_col}' missing from rows with "
            f"{relation_col} == '{relation_value}'."
        )

    grouped = sub.groupby(family_col)
    # A family is "correct" for this relation if *all* of its items are correct.
    correct_by_family = grouped.apply(lambda g: bool((g[gold_col] == g[pred_col]).all()))
    return correct_by_family


def conditional_flip_rates(
    df: pd.DataFrame,
    *,
    gold_col: str = "gold_label",
    pred_col: str = "pred_label",
    family_col: str = "family_id",
    relation_col: str = "variant_relation",
    base_tag: str = "BASE",
    text_flip_tag: str = "TEXT_FLIP",
    image_flip_tag: str = "IMAGE_FLIP",
) -> Dict[str, float]:
    """
    Conditional minimal-edit flip rates over 2×2 families.

    Dataset families are assumed to contain:
      - (I0, q0): BASE
      - (I0, q1): TEXT_FLIP
      - (I1, q0): IMAGE_FLIP
      - (I1, q1): DOUBLE_FLIP   (not used directly in the flip rates)

    Definitions:
      - I_MEFR: among families where BASE is correct, fraction where IMAGE_FLIP
                is also correct.
      - T_MEFR: among families where BASE is correct, fraction where TEXT_FLIP
                is also correct.
      - MEFR  : 0.5 * (I_MEFR + T_MEFR), averaging over defined components.

    Families without a BASE item are ignored. For I_MEFR (resp. T_MEFR), families
    that lack IMAGE_FLIP (resp. TEXT_FLIP) items are ignored for that component.
    """
    base_ok = _family_level_correct(
        df,
        gold_col=gold_col,
        pred_col=pred_col,
        family_col=family_col,
        relation_col=relation_col,
        relation_value=base_tag,
    )

    if base_ok.empty:
        return {"I_MEFR": float("nan"), "T_MEFR": float("nan"), "MEFR": float("nan")}

    text_ok = _family_level_correct(
        df,
        gold_col=gold_col,
        pred_col=pred_col,
        family_col=family_col,
        relation_col=relation_col,
        relation_value=text_flip_tag,
    )
    image_ok = _family_level_correct(
        df,
        gold_col=gold_col,
        pred_col=pred_col,
        family_col=family_col,
        relation_col=relation_col,
        relation_value=image_flip_tag,
    )

    # Restrict to families where BASE is correct.
    base_correct_families = base_ok[base_ok].index

    def _conditional_rate(child_ok: pd.Series) -> float:
        if child_ok.empty:
            return float("nan")
        # Families with at least one child of this relation and base correct.
        eligible = child_ok.loc[child_ok.index.intersection(base_correct_families)]
        if eligible.empty:
            return float("nan")
        return float(eligible.mean())

    i_mefr = _conditional_rate(image_ok)
    t_mefr = _conditional_rate(text_ok)

    vals = [v for v in (i_mefr, t_mefr) if np.isfinite(v)]
    if not vals:
        mefr = float("nan")
    elif len(vals) == 1:
        mefr = float(vals[0])
    else:
        mefr = 0.5 * (i_mefr + t_mefr)  # type: ignore[operator]

    return {"I_MEFR": i_mefr, "T_MEFR": t_mefr, "MEFR": mefr}


def _selective_accuracy_auc(
    gold: Iterable[str],
    pred: Iterable[str],
    confidence: Iterable[float],
) -> float:
    """
    Compute area under the selective-accuracy vs coverage curve.

    Procedure:
      1) Sort examples by confidence descending.
      2) For each prefix k (coverage k/N), compute accuracy over that prefix.
      3) Integrate accuracy as a function of coverage via the trapezoidal rule.

    Returns a value in [0, 1]. Higher is better.
    """
    g = _ensure_numpy(gold)
    p = _ensure_numpy(pred)
    c = _ensure_numpy(confidence).astype(float)

    if g.shape != p.shape or g.shape != c.shape:
        raise ValueError("gold, pred, and confidence must have the same shape.")

    n = len(g)
    if n == 0:
        return float("nan")

    order = np.argsort(-c)  # descending
    g = g[order]
    p = p[order]

    correct = (g == p).astype(float)

    coverages: List[float] = []
    accuracies: List[float] = []

    running_correct = 0.0
    for k in range(1, n + 1):
        running_correct += correct[k - 1]
        cov = k / n
        acc = running_correct / k
        coverages.append(cov)
        accuracies.append(acc)

    x = np.asarray(coverages)
    y = np.asarray(accuracies)
    auc = float(np.trapz(y, x=x))
    return auc


def selrank(
    gold: Iterable[str],
    pred: Iterable[str],
    confidence: Iterable[float],
) -> float:
    """
    Selective Ranking Score (SelRank) - abstention quality metric.

    We work with the *selective accuracy* curve A(cov) and define:

      - A_model  = area under A(cov) for the given confidences.
      - p        = overall accuracy if you never abstain.
      - A_random = p          (random ranking yields flat curve at p).
      - A_oracle = 1.0        (oracle ranking achieves accuracy 1 until all
                               correct items are included).

    Normalized score:

        score = (A_model - A_random) / (A_oracle - A_random)
              = (A_model - p) / (1 - p)      if p < 1
              = 1.0                          if p == 1

    This yields:
      - score = 0 for random ranking,
      - score = 1 for oracle ranking,
      - score in (0, 1) otherwise.
    """
    g = _ensure_numpy(gold)
    p = _ensure_numpy(pred)

    if g.shape != p.shape:
        raise ValueError("gold and pred must have the same shape.")

    if g.size == 0:
        return float("nan")

    # Overall accuracy with no abstention
    baseline_acc = float((g == p).mean())
    if baseline_acc == 1.0:
        return 1.0

    a_model = _selective_accuracy_auc(g, p, confidence)

    # If A_model is NaN, just propagate
    if not np.isfinite(a_model):
        return float("nan")

    # The normalized quantity can be negative if the model's confidence ranking
    # is worse than random (miscalibration). Keep negative values to capture this.
    score = float((a_model - baseline_acc) / max(1e-8, (1.0 - baseline_acc)))
    return float(min(1.0, score))  # Only cap upper bound


def selrank_answered_only(
    gold: Iterable[str],
    pred: Iterable[Optional[str]],
    confidence: Iterable[Optional[float]],
) -> float:
    """
    SelRank computed *only over answered items*.

    "Answered" means pred ∈ {"VISIBLE", "NOT_VISIBLE"}.
    "Abstain" (ABSTAIN) items are excluded entirely from the curve.
    """
    g = _ensure_numpy(gold)
    p = _ensure_numpy(pred).astype(object)
    c = _ensure_numpy(confidence).astype(float)

    if g.shape != p.shape or g.shape != c.shape:
        raise ValueError("gold, pred, and confidence must have the same shape.")

    answered_mask = np.array(
        [
            (x is not None) and (str(x).strip().upper() in {"VISIBLE", "NOT_VISIBLE"})
            for x in p
        ],
        dtype=bool,
    )
    if answered_mask.sum() == 0:
        return float("nan")

    return selrank(g[answered_mask], p[answered_mask], c[answered_mask])


def selrank_answered_only_diagnostics(
    gold: Iterable[str],
    pred: Iterable[Optional[str]],
    confidence: Iterable[Optional[float]],
) -> Dict[str, float]:
    """
    Diagnostics for answered-only SelRank.

    Returns a dict with:
      - n_answered
      - answered_accuracy (p)
      - A_model (area under selective-accuracy curve)
      - raw_normalized (pre-clamp)
      - normalized (post-clamp, equals selrank_answered_only)
    """
    g = _ensure_numpy(gold)
    p = _ensure_numpy(pred).astype(object)
    c = _ensure_numpy(confidence).astype(float)

    if g.shape != p.shape or g.shape != c.shape:
        raise ValueError("gold, pred, and confidence must have the same shape.")

    answered_mask = np.array(
        [
            (x is not None) and (str(x).strip().upper() in {"VISIBLE", "NOT_VISIBLE"})
            for x in p
        ],
        dtype=bool,
    )
    n_answered = int(answered_mask.sum())
    if n_answered == 0:
        return {
            "n_answered": 0.0,
            "answered_accuracy": float("nan"),
            "A_model": float("nan"),
            "raw_normalized": float("nan"),
            "normalized": float("nan"),
        }

    gg = g[answered_mask]
    pp = p[answered_mask]
    cc = c[answered_mask]

    answered_accuracy = float((gg == pp).mean())
    a_model = _selective_accuracy_auc(gg, pp, cc)

    if answered_accuracy == 1.0:
        raw_norm = 1.0
    else:
        raw_norm = float((a_model - answered_accuracy) / max(1e-8, (1.0 - answered_accuracy)))

    norm = float(min(1.0, raw_norm))  # Allow negative values to indicate miscalibration

    return {
        "n_answered": float(n_answered),
        "answered_accuracy": float(answered_accuracy),
        "A_model": float(a_model),
        "raw_normalized": float(raw_norm),
        "normalized": float(norm),
    }


def xor_family_metrics(
    df: pd.DataFrame,
    *,
    gold_col: str = "gold_label",
    pred_col: str = "pred_label",
    family_col: str = "family_id",
    relation_col: str = "variant_relation",
    base_tag: str = "BASE",
    text_flip_tag: str = "TEXT_FLIP",
    image_flip_tag: str = "IMAGE_FLIP",
    double_flip_tag: str = "DOUBLE_FLIP",
) -> Dict[str, float]:
    """
    XOR family metrics:
      - I_MEFR, T_MEFR, MEFR as conditional rates (BASE-correct conditioning)
      - DFAcc: conditional DOUBLE_FLIP correctness among BASE-correct families

    UNDETERMINABLE predictions count as incorrect (since gold is binary).
    """
    flip = conditional_flip_rates(
        df,
        gold_col=gold_col,
        pred_col=pred_col,
        family_col=family_col,
        relation_col=relation_col,
        base_tag=base_tag,
        text_flip_tag=text_flip_tag,
        image_flip_tag=image_flip_tag,
    )

    base_ok = _family_level_correct(
        df,
        gold_col=gold_col,
        pred_col=pred_col,
        family_col=family_col,
        relation_col=relation_col,
        relation_value=base_tag,
    )
    double_ok = _family_level_correct(
        df,
        gold_col=gold_col,
        pred_col=pred_col,
        family_col=family_col,
        relation_col=relation_col,
        relation_value=double_flip_tag,
    )
    if base_ok.empty or double_ok.empty:
        dfacc = float("nan")
    else:
        eligible = double_ok.loc[double_ok.index.intersection(base_ok[base_ok].index)]
        dfacc = float(eligible.mean()) if not eligible.empty else float("nan")

    return {**flip, "DFAcc": dfacc}


def second_order_tom_accuracy(
    df: pd.DataFrame,
    *,
    gold_col: str = "gold_label",
    pred_col: str = "pred_label",
    tom_flag_col: str = "is_second_order_tom",
) -> float:
    """
    Accuracy on second-order ToM subset (E).

    We assume `tom_flag_col` identifies the subset (True/1); the rest of the
    DataFrame is ignored for this metric.
    """
    # If we don't have a ToM flag column yet, treat this metric as undefined.
    if tom_flag_col not in df.columns:
        return float("nan")

    sub = df[df[tom_flag_col].astype(bool)]
    if sub.empty:
        return float("nan")

    if gold_col not in sub.columns or pred_col not in sub.columns:
        raise ValueError(f"gold_col '{gold_col}' or pred_col '{pred_col}' missing from ToM subset.")

    acc = (sub[gold_col] == sub[pred_col]).mean()
    return float(acc)


def invariance_consistency_strict(
    df: pd.DataFrame,
    *,
    gold_col: str = "gold_label",
    pred_col: str = "pred_label",
    family_col: str = "family_id",
    relation_col: str = "variant_relation",
    base_tag: str = "BASE",
    invariant_tag: str = "INVARIANT",
) -> float:
    """
    Strict invariance consistency (IC_strict) between BASE and INVARIANT items.

    For each family:
      - Require at least one BASE and one INVARIANT item.
      - Require that all BASE and INVARIANT items share the same gold label.
      - Condition on all BASE items being predicted correctly.
      - IC_strict is the fraction of such eligible families where *all*
        INVARIANT items are also predicted correctly.

    Families not meeting these criteria are ignored. Returns NaN if there are
    no eligible families.
    """
    required_cols = [gold_col, pred_col, family_col, relation_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        # If we lack the structural metadata entirely, treat as undefined.
        return float("nan")

    grouped = df.groupby(family_col)
    num_families = 0
    num_consistent = 0

    for _, g in grouped:
        base_rows = g[g[relation_col] == base_tag]
        inv_rows = g[g[relation_col] == invariant_tag]
        if base_rows.empty or inv_rows.empty:
            continue

        base_gold = base_rows[gold_col].iloc[0]
        if not (base_rows[gold_col] == base_gold).all():
            continue
        if not (inv_rows[gold_col] == base_gold).all():
            continue

        # Condition on all BASE items being correct.
        base_correct = (base_rows[gold_col] == base_rows[pred_col]).all()
        if not bool(base_correct):
            continue

        num_families += 1
        inv_correct = (inv_rows[gold_col] == inv_rows[pred_col]).all()
        if bool(inv_correct):
            num_consistent += 1

    if num_families == 0:
        return float("nan")

    return float(num_consistent / num_families)


def compute_all_metrics(
    df: pd.DataFrame,
    *,
    gold_col: str = "gold_label",
    pred_col: str = "pred_label",
    confidence_col: str = "confidence",
    family_col: str = "family_id",
    relation_col: str = "variant_relation",
    tom_flag_col: str = "is_second_order_tom",
    family_metrics_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Convenience helper to compute all scalar metrics used in the final composite.

    Returns a dict containing:
      - core_confidence_accuracy   (CAA)
      - I_MEFR, T_MEFR, MEFR       (conditional flip rates)
      - abstention_quality         (SelRank)
      - second_order_accuracy      (ToM subset accuracy)
      - ic_strict                  (BASE vs INVARIANT consistency; not in composite)
    """
    missing = [c for c in (gold_col, pred_col, confidence_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    gold = df[gold_col].values
    pred = df[pred_col].values
    conf = df[confidence_col].astype(float).values

    # XOR-style CAA: treat UNDETERMINABLE as abstain with fixed alpha credit.
    core = caa_with_abstain_penalty(gold, pred, conf, alpha=0.25)

    fam_df = family_metrics_df if family_metrics_df is not None else df

    # Family XOR metrics may be NaN if metadata is absent.
    flip_dict = xor_family_metrics(
        fam_df,
        gold_col=gold_col,
        pred_col=pred_col,
        family_col=family_col,
        relation_col=relation_col,
    )

    # SelRank computed over answered-only items; abstentions are excluded from coverage.
    abst = selrank_answered_only(gold, pred, conf)

    tom_acc = second_order_tom_accuracy(
        df,
        gold_col=gold_col,
        pred_col=pred_col,
        tom_flag_col=tom_flag_col,
    )
    ic_strict = invariance_consistency_strict(
        fam_df,
        gold_col=gold_col,
        pred_col=pred_col,
        family_col=family_col,
        relation_col=relation_col,
    )

    return {
        "core_confidence_accuracy": core,
        "I_MEFR": flip_dict["I_MEFR"],
        "T_MEFR": flip_dict["T_MEFR"],
        "MEFR": flip_dict["MEFR"],
        "DFAcc": flip_dict["DFAcc"],
        "abstention_quality": abst,
        "second_order_accuracy": tom_acc,
        "ic_strict": ic_strict,
    }


def composite_score(
    metrics: Dict[str, float],
    *,
    w_core: float = 0.70,
    w_mefr: float = 0.15,
    w_abst: float = 0.10,
    w_tom: float = 0.05,
) -> float:
    """
    Final composite score based on the provided metric dictionary.

    Default weights follow the current spec:
      - 70% × Confidence-Aware Accuracy (CAA)
      - 15% × Mean conditional flip rate (MEFR = 0.5 × (I_MEFR + T_MEFR))
      - 10% × Abstention Quality (SelRank)
      - 5%  × Second-order ToM accuracy

    IC_strict is reported separately and *not* included in this composite.
    """
    core = metrics.get("core_confidence_accuracy", float("nan"))
    mefr = metrics.get("MEFR", float("nan"))
    abst = metrics.get("abstention_quality", float("nan"))
    tom = metrics.get("second_order_accuracy", float("nan"))

    # If some components are NaN (e.g. no FLIP or no ToM items),
    # we drop their weight and renormalize the remaining weights.
    comps = [core, mefr, abst, tom]
    weights = [w_core, w_mefr, w_abst, w_tom]

    total_w = 0.0
    acc = 0.0
    for val, w in zip(comps, weights):
        if np.isfinite(val):
            acc += w * val
            total_w += w

    if total_w == 0.0:
        return float("nan")

    return float(acc / total_w)


def composite_score_with_effective_weights(
    metrics: Dict[str, float],
    *,
    w_core: float = 0.70,
    w_mefr: float = 0.15,
    w_abst: float = 0.10,
    w_tom: float = 0.05,
) -> tuple[float, Dict[str, float]]:
    """
    Like `composite_score`, but also returns the *effective* weights after dropping NaN
    components and renormalizing.

    Returns:
      (final_score, effective_weight_dict)
    where effective_weight_dict has keys:
      - core_confidence_accuracy
      - MEFR
      - abstention_quality
      - second_order_accuracy
    """
    core = metrics.get("core_confidence_accuracy", float("nan"))
    mefr = metrics.get("MEFR", float("nan"))
    abst = metrics.get("abstention_quality", float("nan"))
    tom = metrics.get("second_order_accuracy", float("nan"))

    comps = [core, mefr, abst, tom]
    keys = ["core_confidence_accuracy", "MEFR", "abstention_quality", "second_order_accuracy"]
    weights = [w_core, w_mefr, w_abst, w_tom]

    total_w = 0.0
    acc = 0.0
    eff: Dict[str, float] = {k: 0.0 for k in keys}

    for key, val, w in zip(keys, comps, weights):
        if np.isfinite(val):
            acc += w * float(val)
            total_w += w
            eff[key] = w

    if total_w == 0.0:
        return float("nan"), eff

    # Renormalize effective weights to sum to 1 over available components.
    for k in eff:
        eff[k] = float(eff[k] / total_w) if eff[k] > 0 else 0.0

    return float(acc / total_w), eff



