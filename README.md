# VLM Visibility Evaluation

> **Note:** This repository accompanies our paper (draft): [Paper_with_Davis.pdf](./Paper_with_Davis.pdf)

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

You can keep using your existing sheets; you don’t need to rewrite paths by hand.

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
python scripts/local/run_visibility_eval.py --input "Sheets/11-14-25.csv" --model gpt

# Incremental run: only fill in rows where this model's outputs are still empty
python scripts/local/run_visibility_eval.py --input "Sheets/11-14-25.csv" --model gpt --only-missing

# You can also point directly at an Excel sheet
python scripts/local/run_visibility_eval.py --input "Sheets/11-14-25.xlsx" --model gpt
```

Under the hood, `--model` can be:

- `gpt` – OpenAI GPT-4 vision (expects `OPENAI_API_KEY` in your env),
- `gemini` – Google Gemini 1.5 Pro (expects `GEMINI_API_KEY`),
- `llava-onevision`, `qwen-vl`, `idefics2` – three strong local vision models wired up via Hugging Face, or
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

If you’d rather get an Excel file directly, just give `--out` an `.xlsx` path, for example:

```bash
python scripts/local/run_visibility_eval.py --input "Sheets/11-14-25.xlsx" --model gpt --out "Sheets/11-14-25.gpt.vlm.xlsx"
```

### Optional: Google Sheets integration (two-stage)

If your source of truth is a Google Sheet (like:
`https://docs.google.com/spreadsheets/d/1-DxRkbEPVvkjCMKDjl8k3tTkFU2FdQ5pr0w6PEuaCmI/...`),
you can treat the cloud sheet as the front-end and this repo as the engine.

#### Stage 1 – Pull the current sheet down

```bash
python scripts/cloud/pull_from_sheets.py \
  --spreadsheet-id "1-DxRkbEPVvkjCMKDjl8k3tTkFU2FdQ5pr0w6PEuaCmI" \
  --input-sheet "11-14-25"
```

- This will create `Sheets/11-14-25.csv` locally (by default).
- It uses the service-account JSON you pointed to via `GOOGLE_APPLICATION_CREDENTIALS`.

#### Stage 2 – Run eval locally

```bash
python scripts/local/run_visibility_eval.py --input "Sheets/11-14-25.csv" --model gpt
```

#### Stage 2b – Score the run (XOR metrics)

```bash
python scripts/scoring/score_vlm_run.py --input "Sheets/11-14-25.gpt.vlm.csv" --model gpt --side xor
```

Notes:
- Headline metrics are computed over **3 cells per family**: BASE/TEXT_FLIP/IMAGE_FLIP. DOUBLE_FLIP is diagnostic-only.
- CAA uses abstain credit **α = 0.25** for `UNDETERMINABLE_FROM_EVIDENCE`.
- AURC is computed over **answered-only** items (UNDETERMINABLE excluded from coverage).
- ToM items are derived from IDs matching `^MA-` (no extra column needed).
- Use `--strict-3cell` (default) to require the 3 headline cells to be parsable for a family to enter MEFR denominators. You can relax with `--no-strict-3cell`.

#### Stage 3 – Push the results back as a new tab

```bash
python scripts/cloud/push_to_sheets.py \
  --spreadsheet-id "1-DxRkbEPVvkjCMKDjl8k3tTkFU2FdQ5pr0w6PEuaCmI" \
  --input "Sheets/11-14-25.gpt.vlm.csv" \
  --sheet-name "11-14-25.gpt" \
  --with-timestamp
```

- This will create (or overwrite) a tab named something like
  `11-14-25.gpt.2025-11-25_1530` inside the same spreadsheet, containing your original columns plus the four XOR output columns:
  `gpt_I0q0_json`, `gpt_I0q1_json`, `gpt_I1q0_json`, `gpt_I1q1_json`.

That will read your existing Excel sheet, run the eval, and write a new `.xlsx` with all the original columns plus the new model output columns, so you can open it straight in Excel or Google Sheets.

