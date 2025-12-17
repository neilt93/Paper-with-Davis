"""
Scoring utilities for the visibility evaluation dataset.

This package implements:
  - confidence-aware accuracy (core task)
  - minimal-edit flip rate (MEFR) for FLIP variants
  - abstention quality via a normalized AURC-style metric
  - second-order ToM subset accuracy
  - a composite score that combines the above with fixed weights

See `score_dataset.py` for a CLI entrypoint.
"""


