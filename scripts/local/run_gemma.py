"""Run the visibility XOR 2×2 eval with PaliGemma2 (Gemma vision baseline).

NOTE: This model is gated on Hugging Face. You likely need:
  - `huggingface-cli login`
  - and to accept the model terms on HF

We use a stable alias `gemma` so the output columns are predictable.
Writes: <input_basename>.gemma.vlm.csv
"""

from __future__ import annotations

from _run_visibility_eval_common import main_with_model


if __name__ == "__main__":
    main_with_model(
        model_name="gemma",
        description="Run visibility eval (XOR 2x2) using PaliGemma2 (alias: gemma).",
    )


