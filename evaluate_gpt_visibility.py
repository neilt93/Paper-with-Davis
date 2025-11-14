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
        base_gold = row["ba]()_
