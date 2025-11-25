"""Core visibility evaluation logic shared by local and cloud entrypoints.

This module knows how to:
  - normalize questions
  - resolve image paths into your local Images/ folder
  - talk to different VLM backends (OpenAI, Gemini, local HF)
  - fill <model>_base_json / <model>_flip_json for Status == 'Done' rows

Entry scripts like `scripts/local/run_visibility_eval.py` and
`scripts/cloud/*` should call into `run_on_dataframe` rather than
reimplementing the logic.
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd
from PIL import Image


PROMPT_TEMPLATE = """You are given one photo. Decide visibility strictly from pixels; do not guess or use world knowledge. The observer is the camera unless a specific person is named. Normalize open-ended content questions (e.g., “what’s in the box?”) into a visibility judgment: can that content be visually determined from this image right now?

ENTITY GROUNDING
• Pronouns/possessives like “he/his”, “she/her”, “their”, or deictics like “this/that” must be grounded by an explicit in-image cue pointing to the referent (e.g., arrow, circle, bounding box, label text on the image).
• If the referent cannot be uniquely identified from such visual cues, return UNDETERMINABLE_FROM_EVIDENCE with reason_code=INSUFFICIENT_CONTEXT.
• Do NOT infer identity from faces, clothing, or world knowledge; only use explicit on-image disambiguators.

Return exactly one minified JSON object (no prose/markdown/trailing commas) with keys in this order:
{{
"label": "VISIBLE | NOT_VISIBLE | UNDETERMINABLE_FROM_EVIDENCE",
"reason_code": "GAZE_DIRECTION | OCCLUSION | OUT_OF_FRAME | LIGHTING_DISTANCE | INHERENTLY_NONVISUAL | AUGMENTED_VISION_REQUIRED | INSUFFICIENT_CONTEXT | MULTI_AGENT_SECOND_ORDER | NONE",
"confidence": number // 0.0-1.0
}}

Rules:
• If label = VISIBLE, set reason_code = NONE.
• If label ≠ VISIBLE, pick exactly one limiting-factor reason_code.
• Precedence if multiple apply: OCCLUSION > OUT_OF_FRAME > GAZE_DIRECTION > LIGHTING_DISTANCE > AUGMENTED_VISION_REQUIRED > INHERENTLY_NONVISUAL > INSUFFICIENT_CONTEXT > MULTI_AGENT_SECOND_ORDER.
• Transparent/clear glass is non-occluding; frosted/translucent counts as occluding for recognition.
• Dark/unlit scenes prevent recognizing fine details unless illumination is evident.
• Do not infer distance if unspecified; if undecidable, use UNDETERMINABLE_FROM_EVIDENCE.
• Confidence is your probability the label is correct (0.0–1.0).

Question: {question}"""


def normalize_question(text: str) -> str:
    """
    Light normalization of the base_question text:
      - strip whitespace
      - capitalize first character
      - ensure trailing '?'
    """
    t = text.strip()
    if not t:
        return t
    t = t[0].upper() + t[1:]
    if not t.endswith("?"):
        t = t.rstrip(".! ") + "?"
    return t


def resolve_image_path(raw_path: str) -> Optional[str]:
    """
    Prefer the original absolute path if it exists; otherwise try common local
    locations based on just the filename, including ./Images/Images.
    """
    if not isinstance(raw_path, str):
        return None

    cleaned = raw_path.strip().strip("'").strip('"')

    if cleaned and os.path.exists(cleaned):
        return cleaned

    filename = os.path.basename(cleaned)
    if not filename:
        return None

    cwd = os.getcwd()
    candidate_dirs = [
        cwd,
        os.path.join(cwd, "Images"),
        os.path.join(cwd, "Images", "Images"),
        os.path.join(cwd, "Images Sheet Versions"),
    ]

    for d in candidate_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate

    return None


def load_table(path: str) -> pd.DataFrame:
    """
    Load either CSV or Excel based on file extension.
    """
    ext = os.path.splitext(path.lower())[1]
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input extension: {ext}")


@dataclass
class ModelResponse:
    raw_text: str


class BaseAdapter:
    def answer(self, image_path: str, prompt: str) -> ModelResponse:
        raise NotImplementedError


class GPTAdapter(BaseAdapter):
    """
    OpenAI GPT vision (e.g. gpt-4.1 / gpt-4o).
    Requires OPENAI_API_KEY in environment.
    """

    def __init__(self, model: str = "gpt-4.1-mini"):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            raise RuntimeError("Missing openai package. Install with `pip install openai`.")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def answer(self, image_path: str, prompt: str) -> ModelResponse:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        b64 = __import__("base64").b64encode(img_bytes).decode("utf-8")
        msg = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
                ],
            }
        ]
        resp = self.client.responses.create(model=self.model, input=msg)
        text = resp.output[0].content[0].text  # type: ignore[attr-defined]
        return ModelResponse(raw_text=text)


class GeminiAdapter(BaseAdapter):
    """
    Google Gemini 1.5 Pro vision.
    Requires GEMINI_API_KEY in environment and google-generativeai installed.
    """

    def __init__(self, model: str = "gemini-1.5-pro"):
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError:
            raise RuntimeError("Missing google-generativeai. Install with `pip install google-generativeai`.")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment.")
        genai.configure(api_key=api_key)
        self.genai = genai
        self.model = genai.GenerativeModel(model)

    def answer(self, image_path: str, prompt: str) -> ModelResponse:
        img = Image.open(image_path)
        resp = self.model.generate_content([prompt, img])
        return ModelResponse(raw_text=resp.text)


class LocalHFAdapter(BaseAdapter):
    """
    Local Hugging Face vision-language model (e.g. LLaVA, Qwen2.5-VL).
    `model_id` should be a HF repo id.
    """

    def __init__(self, model_id: str):
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM  # type: ignore
            import torch  # type: ignore
        except ImportError:
            raise RuntimeError("Missing transformers/torch. Install with `pip install transformers torch`.")

        self.torch = torch
        self.AutoProcessor = AutoProcessor
        self.AutoModelForCausalLM = AutoModelForCausalLM

        if torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval()

    def answer(self, image_path: str, prompt: str) -> ModelResponse:
        image = Image.open(image_path).convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        tpl = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=tpl, images=image, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=128)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return ModelResponse(raw_text=text.strip())


def get_adapter(model_name: str) -> BaseAdapter:
    """Dispatch based on a simple model name."""
    name = model_name.lower()

    if name in {"gpt", "gpt4", "gpt-4o", "openai"}:
        return GPTAdapter(model="gpt-4.1-mini")

    if name in {"gemini", "gemini-1.5"}:
        return GeminiAdapter(model="gemini-1.5-pro")

    hf_map = {
        "llava-onevision": "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
        "qwen-vl": "Qwen/Qwen2.5-VL-7B-Instruct",
        "idefics2": "HuggingFaceM4/idefics2-8b",
    }
    if name in hf_map:
        return LocalHFAdapter(hf_map[name])

    return LocalHFAdapter(model_name)


def run_on_dataframe(df: pd.DataFrame, model_name: str, only_missing: bool = False) -> pd.DataFrame:
    """
    Core evaluation logic:
      - filter to Status == 'Done'
      - normalize questions
      - resolve image paths
      - call the chosen model adapter
      - fill <model>_base_json / <model>_flip_json

    Returns a new DataFrame with only the 'Done' rows and the new columns attached.
    """
    required_cols = {"id", "Status", "pic_base", "pic_flip", "base_question"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    total_rows = len(df)
    df_done = df[df["Status"] == "Done"].copy()
    done_rows = len(df_done)
    if df_done.empty:
        raise ValueError("No rows with Status == 'Done' in input.")

    print(f"[INFO] Filtering rows: total={total_rows} done={done_rows}", file=sys.stderr)

    df_done = df_done.sort_values("id")

    col_base = f"{model_name}_base_json"
    col_flip = f"{model_name}_flip_json"

    if col_base not in df_done.columns:
        df_done[col_base] = ""
    if col_flip not in df_done.columns:
        df_done[col_flip] = ""

    adapter = get_adapter(model_name)

    processed_rows = 0
    skipped_rows = 0
    skipped_base = 0
    skipped_flip = 0

    for idx, row in df_done.iterrows():
        row_id = str(row["id"])
        raw_q = row["base_question"]
        q_text = "" if not isinstance(raw_q, str) else normalize_question(str(raw_q))

        existing_base = str(row.get(col_base, "") or "")
        existing_flip = str(row.get(col_flip, "") or "")

        if only_missing and existing_base.strip() and existing_flip.strip():
            print(
                f"[SKIP] id={row_id} reason=already_has_{col_base}_and_{col_flip}",
                file=sys.stderr,
            )
            skipped_rows += 1
            continue

        if not q_text:
            print(
                f"[SKIP] id={row_id} reason=missing_or_empty_base_question",
                file=sys.stderr,
            )
            skipped_rows += 1
            continue

        full_prompt = PROMPT_TEMPLATE.format(question=q_text)

        base_name = os.path.basename(str(row["pic_base"]))
        flip_name = os.path.basename(str(row["pic_flip"]))

        print(
            f"[RUN] model={model_name} id={row_id} "
            f"base={base_name} flip={flip_name} "
            f"question={q_text}",
            file=sys.stderr,
        )

        # base
        base_path = resolve_image_path(row["pic_base"])
        if base_path is None:
            print(
                f"[WARN] id={row_id} which=base reason=image_not_found raw_path={row['pic_base']}",
                file=sys.stderr,
            )
            skipped_base += 1
        else:
            try:
                resp = adapter.answer(base_path, full_prompt)
                df_done.at[idx, col_base] = resp.raw_text
            except Exception as e:
                print(
                    f"[ERROR] id={row_id} which=base reason=model_error msg={e}",
                    file=sys.stderr,
                )
                df_done.at[idx, col_base] = f"ERROR: {e}"

        # flip
        flip_path = resolve_image_path(row["pic_flip"])
        if flip_path is None:
            print(
                f"[WARN] id={row_id} which=flip reason=image_not_found raw_path={row['pic_flip']}",
                file=sys.stderr,
            )
            skipped_flip += 1
        else:
            try:
                resp = adapter.answer(flip_path, full_prompt)
                df_done.at[idx, col_flip] = resp.raw_text
            except Exception as e:
                print(
                    f"[ERROR] id={row_id} which=flip reason=model_error msg={e}",
                    file=sys.stderr,
                )
                df_done.at[idx, col_flip] = f"ERROR: {e}"

        processed_rows += 1

    print(
        f"[INFO] Core eval complete: processed_rows={processed_rows} "
        f"skipped_rows={skipped_rows} skipped_base={skipped_base} skipped_flip={skipped_flip}",
        file=sys.stderr,
    )

    return df_done


