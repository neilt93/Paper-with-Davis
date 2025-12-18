"""Run the visibility XOR 2×2 eval with LLaVA-OneVision (local HF).

Writes: <input_basename>.llava-onevision.vlm.csv
"""

from __future__ import annotations

from _run_visibility_eval_common import main_with_model


if __name__ == "__main__":
    main_with_model(
        model_name="llava-onevision",
        description="Run visibility eval (XOR 2x2) using LLaVA-OneVision.",
    )


