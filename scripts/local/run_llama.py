"""Run the visibility XOR 2×2 eval with Llama 3.2 Vision.

NOTE: This model is gated on Hugging Face. You likely need:
  - `huggingface-cli login`
  - and to accept the model terms on HF

We use a stable alias `llama-vision` so the output columns are predictable.
Writes: <input_basename>.llama-vision.vlm.csv
"""

from __future__ import annotations

from _run_visibility_eval_common import main_with_model


if __name__ == "__main__":
    main_with_model(
        model_name="llama-vision",
        description="Run visibility eval (XOR 2x2) using Llama 3.2 Vision (alias: llama-vision).",
    )


