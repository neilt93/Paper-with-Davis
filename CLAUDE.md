# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VLM Visibility Evaluation system for testing Vision Language Models on their ability to determine whether specific visual information is perceivable in images. Uses a 2×2 XOR experimental design.

## Key Commands

### Run inference on a model
```bash
python scripts/local/run_visibility_eval.py --input "Sheets/FINAL_Pictures_DB.csv" --model gpt
python scripts/local/run_visibility_eval.py --input "Sheets/FINAL_Pictures_DB.csv" --model gemini
python scripts/local/run_visibility_eval.py --input "Sheets/FINAL_Pictures_DB.csv" --model claude
```
- Outputs: `<input>.<model>.vlm.csv` with 4 JSON columns per model
- Use `--only-missing` to resume partial runs

### Score a model run
```bash
python scripts/scoring/score_vlm_run.py --input "Sheets/FINAL_Pictures_DB.gpt.vlm.csv" --model gpt --side xor --strict-3cell
```
- Outputs: `*.scored.csv` and prints metrics report

### Local GPU models (requires GPU)
```bash
python scripts/local/run_visibility_eval.py --input "Sheets/FINAL_Pictures_DB.csv" --model llava
python scripts/local/run_visibility_eval.py --input "Sheets/FINAL_Pictures_DB.csv" --model qwen
python scripts/local/run_visibility_eval.py --input "Sheets/FINAL_Pictures_DB.csv" --model llama
```

## Architecture

### XOR 2×2 Design
Each row produces 4 cells:
- `I0q0` (BASE): base_image × base_question → gold: NOT_VISIBLE
- `I0q1` (TEXT_FLIP): base_image × flip_question → gold: VISIBLE
- `I1q0` (IMAGE_FLIP): flip_image × base_question → gold: VISIBLE
- `I1q1` (DOUBLE_FLIP): flip_image × flip_question → gold: NOT_VISIBLE (diagnostic only)

### Core Files
- `scripts/visibility_eval_core.py` — Model adapters (GPTAdapter, GeminiAdapter, ClaudeAdapter, LocalHFAdapter), prompt template, image resolution, `run_on_dataframe()`
- `scripts/scoring/metrics.py` — All metric implementations: CAA, MEFR, AURC, ToMAcc, composite score
- `scripts/scoring/score_vlm_run.py` — CLI for scoring, JSON extraction, report generation

### Metrics (composite = 0.7×CAA + 0.15×MEFR + 0.10×AURC + 0.05×ToMAcc)
- **CAA** (Confidence-Aware Accuracy): α=0.25 penalty for abstentions
- **MEFR** (Mean Estimated Flip Rate): average of I_MEFR and T_MEFR, measures sensitivity to flips
- **AURC** (Abstention Under-Rejection Curve): calibration quality, negative values indicate miscalibration
- **ToMAcc**: Accuracy on rows with ID prefix `MA-` (Theory of Mind items)

### Dataset Structure
Required columns: `ID`, `Status`, `pic_base`, `pic_flip`, `base_question`, `flip_question`

Images resolved from: `./Images/Images/`, `./Images/`, `./Images Sheet Versions/`, `./`

## Environment

API keys in `.env`:
```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
HF_TOKEN=...  # for local models
```

Note: Shell environment variables override `.env` (dotenv doesn't use `override=True`). If a key seems stale, check `echo $OPENAI_API_KEY` etc.

## Important Notes

- Only rows with `Status == "Done"` are evaluated
- Scoring uses 3 cells (BASE/TEXT_FLIP/IMAGE_FLIP); DOUBLE_FLIP is diagnostic only
- Current dataset: `Sheets/FINAL_Pictures_DB.csv` (100 rows, all verified)
- Old scores archived in `Sheets/archive_old_scores/`, new scores in `Sheets/new_scores/`
