"""Run the visibility XOR 2×2 eval with Qwen2.5-VL (local HF).

Default uses the 4-bit bitsandbytes variant for speed on Windows:
  - model alias: qwen-vl-4bit

Writes: <input_basename>.qwen-vl-4bit.vlm.csv
"""

from __future__ import annotations

from _run_visibility_eval_common import main_with_model


if __name__ == "__main__":
    main_with_model(
        model_name="qwen-vl-4bit",
        description="Run visibility eval (XOR 2x2) using Qwen2.5-VL (4-bit).",
    )


