# VB: Visibility Benchmark for Vision-Language Models

This repository contains the dataset, evaluation code, and paper source for **VB** (Visibility Benchmark), a benchmark that tests whether vision-language models can determine what is and is not visible in a photograph---and abstain when a human viewer cannot reliably answer.

## Paper

The paper is located in [`paper/main.tex`](./paper/main.tex) and can be built with `pdflatex`/`bibtex`. It evaluates **9 vision-language models** (3 flagship closed-source, 3 prior-generation closed-source, 3 open-source 8--12B) across 100 families using a 2x2 XOR design (300 headline evaluation cells per model). Key findings:

- **GPT-4o** achieves the best composite score (0.728), followed by Gemini 3.1 Pro (0.727) and Gemini 2.5 Pro (0.678)
- **Gemma 3 12B** is the best open-source model (0.505), surpassing one prior-generation closed-source system
- Text-flip robustness exceeds image-flip robustness for 6/9 models
- Confidence calibration varies substantially across models

Models are scored on confidence-aware accuracy with abstention (CAA), minimal-edit flip rate (MEFR), confidence-ranked selective prediction (SelRank), and second-order perspective reasoning (ToMAcc).

## Installation

```bash
git clone https://github.com/<your-org>/visibility-benchmark.git
cd visibility-benchmark
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then fill in your API keys
```

## Overview

This project evaluates Vision Language Models (VLMs) on their ability to correctly determine whether specific visual information is perceivable in images.

---

## Visibility evaluation scripts

This repo is meant to make it painless to run your visibility benchmark on real models and keep everything in sync with your sheets.

### What lives here

- `scripts/local/run_visibility_eval.py` – the main local entry point. Give it a sheet (CSV or Excel) and a model name (API or local HF), and it will:
  - filter to rows with `Status == "Done"`,
  - normalize image paths and questions,
  - run the model on the **XOR 2×2** (base/flip image × base/flip question), and
  - write per-cell JSON outputs back to a new file.
- `scripts/local/vlm_inference.py` – an earlier, LLaVA-only local inference script (kept around as a backup).
- `scripts/local/evaluate_gpt_visibility.py` – computes GPT-based visibility metrics from a CSV that already has `gpt_output_*` columns (your original GPT eval code).
- `scripts/scoring/score_vlm_run.py` – scores a `*.vlm.csv` produced by `run_visibility_eval.py` using the XOR metric spec (CAA/MEFR/AURC/ToMAcc) and prints a report.

### What your sheet needs to look like

For `scripts/local/run_visibility_eval.py`, the input table should (at minimum) have:

- `ID` (or `id`) – row identifier (e.g. `AV-01`).
- `Status` – only rows with `Status == "Done"` are evaluated.
- `pic_base`, `pic_flip` – image paths as they appear in your workflow. These can be absolute macOS-style paths like `/Users/neiltripathi/Documents/Paper with Davis/Images/IMG_0394.jpeg`.
- `base_question` – the base question.
- `flip_question` – the text-flipped question (typically the strict negation of `base_question`).
- `base_label` – optional. In the XOR dataset we assume BASE is always `NOT_VISIBLE`, so scoring does not require this column.

You can keep using your existing sheets; you don't need to rewrite paths by hand.

#### How image paths are resolved

For each `pic_base` / `pic_flip`, the script:

- strips quotes and whitespace,
- takes just the **filename** (e.g. `IMG_0394.jpeg` from `/Users/.../Images/IMG_0394.jpeg`),
- then looks for that filename under (in order):
  - `./Images/Images/<filename>`
  - `./Images/<filename>`
  - `./Images Sheet Versions/<filename>`
  - `./<filename>`

If none of these exist, it logs a warning like:

```text
[WARN] id=AV-03 which=base reason=image_not_found raw_path=/Users/.../IMG_0682.jpeg'
```

and skips the model call for that side.

#### How questions are normalized

Before sending to the model, `base_question` and `flip_question` are lightly cleaned up:

- leading/trailing whitespace is stripped,
- the first character is capitalized, and
- a trailing `?` is added if missing.

So `is the serial number readable` becomes `Is the serial number readable?` in both the logs and the actual prompt.

### How to run `run_visibility_eval.py`

From the repo root:

```bash
# Full run: evaluate every row with Status == 'Done' for this model
python scripts/local/run_visibility_eval.py --input "Sheets/FINAL_Pictures_DB.csv" --model gpt

# Incremental run: only fill in rows where this model's outputs are still empty
python scripts/local/run_visibility_eval.py --input "Sheets/FINAL_Pictures_DB.csv" --model gpt --only-missing
```

Under the hood, `--model` can be:

- `gpt` – OpenAI GPT-4 vision (expects `OPENAI_API_KEY` in your env),
- `gemini` – Google Gemini Pro (expects `GEMINI_API_KEY`),
- `claude` – Anthropic Claude (expects `ANTHROPIC_API_KEY`),
- `llava`, `qwen`, `llama` – local vision models via Hugging Face (expects `HF_TOKEN`), or
- any valid HF repo id for a local vision model.

### What you get back

By default, the script writes a **CSV** next to the input, named:

- `<input_basename>.<model>.vlm.csv`

and adds four columns (the XOR 2×2):

- `<model>_I0q0_json` (BASE): `pic_base × base_question`
- `<model>_I0q1_json` (TEXT_FLIP): `pic_base × flip_question`
- `<model>_I1q0_json` (IMAGE_FLIP): `pic_flip × base_question`
- `<model>_I1q1_json` (DOUBLE_FLIP, diagnostic): `pic_flip × flip_question`

Each cell contains the raw JSON/text from the model for that (image, question) pair, or an `ERROR: ...` message if something went wrong on that row.

### Scoring

```bash
python scripts/scoring/score_vlm_run.py --input "Sheets/FINAL_Pictures_DB.gpt.vlm.csv" --model gpt --side xor --strict-3cell
```

Notes:
- Headline metrics are computed over **3 cells per family**: BASE/TEXT_FLIP/IMAGE_FLIP. DOUBLE_FLIP is diagnostic-only.
- CAA uses abstain credit **α = 0.25** for `UNDETERMINABLE_FROM_EVIDENCE`.
- AURC is computed over **answered-only** items (UNDETERMINABLE excluded from coverage).
- ToM items are derived from IDs matching `^MA-` (no extra column needed).
- Use `--strict-3cell` (default) to require the 3 headline cells to be parsable for a family to enter MEFR denominators. You can relax with `--no-strict-3cell`.

## Citation

If you use the VB benchmark in your research, please cite:

```bibtex
@article{tripathi_vb_2026,
  author  = {Neil Tripathi},
  title   = {{VB}: Visibility Benchmark for Visibility and Perspective Reasoning in Images},
  year    = {2026},
  note    = {Available at \url{https://github.com/<your-org>/visibility-benchmark}}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
