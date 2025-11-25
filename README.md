## Visibility evaluation scripts

Working scripts in this repo:

- `run_visibility_eval.py` – unified runner that takes a CSV + model name (API or local HF) and generates per-image JSON predictions.
- `vlm_inference.py` – earlier LLaVA-only local inference script.
- `evaluate_gpt_visibility.py` – compute GPT-based visibility metrics from a CSV that already has `gpt_output_*` columns.

### Expected CSV format for `run_visibility_eval.py`

The input CSV (e.g. `Sheets/11-14-25.csv`) should have at least:

- `id` – row identifier (e.g. `AV-01`).
- `Status` – only rows with `Status == "Done"` are evaluated.
- `pic_base`, `pic_flip` – original image paths (can be absolute like `/Users/.../Images/IMG_0394.jpeg`).
- `base_question` – the natural-language question for that pair.

**Image path preprocessing**:

- For each `pic_base` / `pic_flip`, the script:
  - strips quotes and whitespace,
  - takes just the **filename** (e.g. `IMG_0394.jpeg` from `/Users/.../Images/IMG_0394.jpeg`),
  - looks for the file under (in order):
    - `./Images/Images/<filename>`
    - `./Images/<filename>`
    - `./Images Sheet Versions/<filename>`
    - `./<filename>`
- This lets you keep absolute macOS-style paths in the sheet while running the eval on Windows; only the filename matters.

### Running `run_visibility_eval.py`

From the repo root:

```bash
# You can point directly at either CSV or Excel:
python run_visibility_eval.py --input "Sheets/11-14-25.csv" --model gpt
python run_visibility_eval.py --input "Sheets/11-14-25.xlsx" --model gpt

# API models (need env vars like OPENAI_API_KEY / GEMINI_API_KEY)
python run_visibility_eval.py --input "Sheets/11-14-25.csv" --model gpt
python run_visibility_eval.py --input "Sheets/11-14-25.csv" --model gemini

# Local HF models (either friendly name or raw repo id)
python run_visibility_eval.py --input "Sheets/11-14-25.csv" --model llava-onevision
python run_visibility_eval.py --input "Sheets/11-14-25.csv" --model qwen-vl
python run_visibility_eval.py --input "Sheets/11-14-25.csv" --model idefics2
```

By default, the script creates a **CSV** next to the input, named `<input_basename>.<model>.vlm.csv`, with two new columns:

- `<model>_base_json`
- `<model>_flip_json`

If you want an Excel file instead, pass an `.xlsx` path to `--out`, for example:

```bash
python run_visibility_eval.py --input "Sheets/11-14-25.xlsx" --model gpt --out "Sheets/11-14-25.gpt.vlm.xlsx"
```


