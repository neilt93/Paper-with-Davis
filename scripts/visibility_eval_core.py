"""Core visibility evaluation logic shared by local and cloud entrypoints.

This module knows how to:
  - normalize questions
  - resolve image paths into your local Images/ folder
  - talk to different VLM backends (OpenAI, Gemini, local HF)
  - fill per-model output columns for Status == 'Done' rows, including:
        <model>_base_json / <model>_flip_json
        <model>_output_base / <model>_output_flip

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

# Optional: load environment variables from a local .env file so that
# OPENAI_API_KEY, GEMINI_API_KEY, HF tokens, etc. can be set there
# without needing to export them in the shell.
try:  # pragma: no cover - best-effort convenience only
    from dotenv import load_dotenv  # type: ignore
except Exception:  # ImportError or anything else
    load_dotenv = None  # type: ignore

if load_dotenv is not None:
    # Prefer a .env in the project root (current working directory).
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        # Fallback: let python-dotenv search default locations.
        load_dotenv()


PROMPT_TEMPLATE = """You are given one photo and one question. Decide whether the question’s claim is true using only pixels in this photo. Do not guess and do not use world knowledge.

Observer

The observer is the camera unless the question explicitly names a person in the image as the observer.

How to interpret questions

Treat questions like “Is X visible/readable/clear?” or “Is person A looking at X?” as a true/false claim you must judge from the image.

For open-ended content questions (eg “what’s in the box?”, “what time is it?”), convert them into: “Is that information visually determinable from this photo right now?”

Referring expressions (who is “he/she/they/it”)

Do not require arrows, circles, or any on-image markings.

If the question uses pronouns or vague references, you must be able to identify the referent unambiguously from the image plus the question text alone.

Acceptable disambiguation: only one plausible referent exists, or the question uniquely specifies it via clear visual descriptors (eg “the man on the left”, “the woman in the red jacket”, “the cat on the sofa”).

If multiple referents are plausible and the text does not uniquely pick one, choose ABSTAIN with reason_code=INSUFFICIENT_CONTEXT.

Output format
Return exactly one minified JSON object (no prose, no markdown, no trailing commas) with keys in this order:
{{"label":"VISIBLE|NOT_VISIBLE|ABSTAIN","reason_code":"GAZE_DIRECTION|OCCLUSION|OUT_OF_FRAME|LIGHTING_DISTANCE|INHERENTLY_NONVISUAL|AUGMENTED_VISION_REQUIRED|INSUFFICIENT_CONTEXT|MULTI_AGENT_SECOND_ORDER|NONE","confidence":number}}

Label meanings

VISIBLE: the claim in the question is clearly true from pixels.

NOT_VISIBLE: the claim in the question is clearly false from pixels.

ABSTAIN: you cannot decide true vs false with reasonable confidence from this image.

Reason codes

If label="VISIBLE", set reason_code="NONE".

If label="ABSTAIN", pick exactly one reason_code explaining what prevents a decision.

If label="NOT_VISIBLE":

If the claim is false because the opposite is clearly true (eg the question asserts “not visible” but it is plainly visible), you may set reason_code="NONE".

Otherwise pick exactly one limiting-factor reason_code explaining why the claim is false.

Precedence if multiple apply:
OCCLUSION > OUT_OF_FRAME > GAZE_DIRECTION > LIGHTING_DISTANCE > AUGMENTED_VISION_REQUIRED > INHERENTLY_NONVISUAL > INSUFFICIENT_CONTEXT > MULTI_AGENT_SECOND_ORDER

Transparent clear glass is non-occluding; frosted/translucent counts as occluding for recognition.

Confidence

Confidence is your probability that your chosen label is correct (0.0 to 1.0).

Use your own internal thresholding. If you cannot decide with reasonable confidence, choose ABSTAIN.

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
        """
        Generic local HF VLM loader.

        For most chat-style VLMs (e.g. LLaVA-OneVision) we can use
        `AutoModelForCausalLM`. For Qwen2.5-VL, we instead need
        `AutoModelForImageTextToText` to handle its multimodal config.
        """
        try:
            from transformers import (  # type: ignore
                AutoProcessor,
                AutoModelForCausalLM,
            )
            try:
                # Newer multimodal models like Qwen2.5-VL use this head
                from transformers import AutoModelForImageTextToText  # type: ignore
            except ImportError:
                AutoModelForImageTextToText = None  # type: ignore[assignment]
            import torch  # type: ignore
        except ImportError:
            raise RuntimeError("Missing transformers/torch. Install with `pip install transformers torch`.")

        self.torch = torch
        self.AutoProcessor = AutoProcessor
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoModelForImageTextToText = AutoModelForImageTextToText  # type: ignore[attr-defined]

        if torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        # Special-case Qwen2.5-VL to avoid the AutoModelForCausalLM config error
        is_qwen_vision = "qwen2.5-vl" in model_id.lower() or "qwen2_5_vl" in model_id.lower()

        if is_qwen_vision and self.AutoModelForImageTextToText is not None:  # type: ignore[truthy-function]
            ModelCls = self.AutoModelForImageTextToText
        else:
            ModelCls = self.AutoModelForCausalLM

        self.model = ModelCls.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval()

    def answer(self, image_path: str, prompt: str) -> ModelResponse:
        """
        Run a single image+prompt through the local HF VLM.

        To control memory usage (especially for large vision backbones like
        Qwen2.5-VL), we downscale the image so that its longest side is
        at most 768 pixels before passing it to the processor. The processor
        may further resize as required by the model config.
        """
        image = Image.open(image_path).convert("RGB")

        # Light downscaling to reduce VRAM usage without killing detail.
        max_side = 768
        w, h = image.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            new_size = (int(w * scale), int(h * scale))
            image = image.resize(new_size, Image.LANCZOS)

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


class QwenLocalAdapter(BaseAdapter):
    """
    Thin wrapper around LocalHFAdapter for Qwen2.5-VL.

    Qwen's chat template often returns the full conversation transcript
    (system/user/assistant) in the decoded text. For scoring, we want the
    final JSON object only, so this adapter:
      - prefers running on GPU (float32) when available for speed
      - falls back to CPU when CUDA is not available
      - post-processes the raw text and keeps just the last {...} block
        if it parses as JSON.
    """

    def __init__(self, model_id: str):
        try:
            from transformers import (  # type: ignore
                AutoProcessor,
                AutoModelForCausalLM,
            )
            try:
                from transformers import AutoModelForImageTextToText  # type: ignore
            except ImportError:
                AutoModelForImageTextToText = None  # type: ignore[assignment]
            import torch  # type: ignore
        except ImportError:
            raise RuntimeError("Missing transformers/torch. Install with `pip install transformers torch`.")

        self.torch = torch
        self.AutoProcessor = AutoProcessor
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoModelForImageTextToText = AutoModelForImageTextToText  # type: ignore[attr-defined]

        # Prefer GPU if available; use float32 for extra stability.
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        is_qwen_vision = "qwen2.5-vl" in model_id.lower() or "qwen2_5_vl" in model_id.lower()
        if is_qwen_vision and self.AutoModelForImageTextToText is not None:  # type: ignore[truthy-function]
            ModelCls = self.AutoModelForImageTextToText
        else:
            ModelCls = self.AutoModelForCausalLM

        self.model = ModelCls.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        # Move the whole model to the chosen device (GPU if available).
        self.model.to(device)
        self.model.eval()

    def answer(self, image_path: str, prompt: str) -> ModelResponse:
        """
        Run a single image+prompt through Qwen2.5-VL and return a cleaned
        JSON string if possible.
        """
        image = Image.open(image_path).convert("RGB")

        # Slight downscaling for speed on CPU; keep enough detail for Qwen.
        max_side = 768
        w, h = image.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            new_size = (int(w * scale), int(h * scale))
            image = image.resize(new_size, Image.LANCZOS)

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

        # Post-process to keep only the final JSON object if we can find it.
        s = text.strip()
        s = text.strip()
        try:
            # Take the last {...} block in the string.
            start = s.rindex("{")
            end = s.rindex("}") + 1
            candidate = s[start:end]

            import json  # local import to avoid top-level dependency if unused

            json.loads(candidate)
            return ModelResponse(raw_text=candidate)
        except Exception:
            # Fall back to the full decoded text if we can't isolate valid JSON.
            return ModelResponse(raw_text=text.strip())

def get_adapter(model_name: str) -> BaseAdapter:
    """Dispatch based on a simple model name."""
    name = model_name.lower()

    if name in {"gpt", "gpt4", "gpt-4o", "openai"}:
        return GPTAdapter(model="gpt-4.1-mini")

    if name in {"gemini", "gemini-1.5"}:
        return GeminiAdapter(model="gemini-1.5-pro")

    # Qwen gets a custom adapter so we can post-process its outputs.
    if name == "qwen-vl":
        return QwenLocalAdapter("Qwen/Qwen2.5-VL-3B-Instruct")

    hf_map = {
        "llava-onevision": "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
        "idefics2": "HuggingFaceM4/idefics2-8b",
    }
    if name in hf_map:
        return LocalHFAdapter(hf_map[name])

    return LocalHFAdapter(model_name)


def run_on_dataframe(df: pd.DataFrame, model_name: str, only_missing: bool = False) -> pd.DataFrame:
    """
    Core evaluation logic:
      - verify that required "base" columns are present
      - filter to Status == 'Done' for running the model
      - normalize questions
      - resolve image paths
      - call the chosen model adapter
      - fill per-model 2×2 columns:
          <model>_I0q0_json / <model>_I0q1_json / <model>_I1q0_json / <model>_I1q1_json

    Returns a new DataFrame that preserves all original rows/columns and
    appends the per-model output columns at the end. Rows with Status != 'Done'
    are left untouched.
    """
    # ------------------------------------------------------------------
    # Schema normalization for newer sheet layouts
    # ------------------------------------------------------------------
    # Newer sheets use a compact header, e.g.:
    #   ID, Status, base_question, base_setup, flip_question, flip_change,
    #   pic_base, pic_flip, indoor_outdoor
    #
    # We normalize this into the internal schema expected by the XOR 2×2 eval:
    #   - ensure 'id' exists (mapping from 'ID' if necessary)
    df = df.copy()
    if "id" not in df.columns and "ID" in df.columns:
        df["id"] = df["ID"]

    # "Base" schema we expect from the Google Sheet after normalization.
    required_cols = {
        "id",
        "Status",
        "base_question",
        "base_setup",
        "flip_question",
        "flip_change",
        "pic_base",
        "pic_flip",
        "indoor_outdoor",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    total_rows = len(df)
    df_out = df.copy()

    # We only *run* the model on rows marked Done, but keep all rows in output.
    mask_done = df_out["Status"] == "Done"
    done_rows = int(mask_done.sum())
    if done_rows == 0:
        raise ValueError("No rows with Status == 'Done' in input.")

    print(f"[INFO] Filtering rows: total={total_rows} done={done_rows}", file=sys.stderr)

    # Column names for this model (XOR-style 2×2).
    col_i0q0_json = f"{model_name}_I0q0_json"
    col_i0q1_json = f"{model_name}_I0q1_json"
    col_i1q0_json = f"{model_name}_I1q0_json"
    col_i1q1_json = f"{model_name}_I1q1_json"

    # Ensure the per-model columns exist; new ones will be appended at the end.
    for col in (
        col_i0q0_json,
        col_i0q1_json,
        col_i1q0_json,
        col_i1q1_json,
    ):
        if col not in df_out.columns:
            df_out[col] = ""

    adapter = get_adapter(model_name)

    processed_rows = 0
    skipped_rows = 0
    skipped_base = 0
    skipped_flip = 0

    # Iterate over the Done rows in id order for reproducibility.
    done_indices = df_out[mask_done].sort_values("id").index

    for idx in done_indices:
        row = df_out.loc[idx]
        row_id = str(row["id"])
        raw_q0 = row["base_question"]
        q0_text = "" if not isinstance(raw_q0, str) else normalize_question(str(raw_q0))
        raw_q1 = row.get("flip_question", "")
        q1_text = "" if not isinstance(raw_q1, str) else normalize_question(str(raw_q1))

        existing_base = str(row.get(col_i0q0_json, "") or "")
        existing_flip = str(row.get(col_i1q0_json, "") or "")
        existing_i0q1 = str(row.get(col_i0q1_json, "") or "")
        existing_i1q1 = str(row.get(col_i1q1_json, "") or "")

        if only_missing and existing_base.strip() and existing_flip.strip() and existing_i0q1.strip() and existing_i1q1.strip():
            print(
                f"[SKIP] id={row_id} reason=already_has_2x2_outputs",
                file=sys.stderr,
            )
            skipped_rows += 1
            continue

        if not q0_text:
            print(
                f"[SKIP] id={row_id} reason=missing_or_empty_base_question",
                file=sys.stderr,
            )
            skipped_rows += 1
            continue
        if not q1_text:
            print(
                f"[SKIP] id={row_id} reason=missing_or_empty_flip_question",
                file=sys.stderr,
            )
            skipped_rows += 1
            continue

        prompt_q0 = PROMPT_TEMPLATE.format(question=q0_text)
        prompt_q1 = PROMPT_TEMPLATE.format(question=q1_text)

        base_name = os.path.basename(str(row["pic_base"]))
        flip_name = os.path.basename(str(row["pic_flip"]))

        print(
            f"[RUN] model={model_name} id={row_id} "
            f"base={base_name} flip={flip_name} "
            f"q0={q0_text} q1={q1_text}",
            file=sys.stderr,
        )

        # Resolve image paths once.
        base_path = resolve_image_path(row["pic_base"])
        flip_path = resolve_image_path(row["pic_flip"])
        if base_path is None:
            print(
                f"[WARN] id={row_id} which=base reason=image_not_found raw_path={row['pic_base']}",
                file=sys.stderr,
            )
            skipped_base += 1
        if flip_path is None:
            print(
                f"[WARN] id={row_id} which=flip reason=image_not_found raw_path={row['pic_flip']}",
                file=sys.stderr,
            )
            skipped_flip += 1

        def _run(image_path: Optional[str], prompt: str, which: str) -> str:
            if image_path is None:
                return ""
            try:
                resp = adapter.answer(image_path, prompt)
                return resp.raw_text
            except Exception as e:
                print(
                    f"[ERROR] id={row_id} which={which} reason=model_error msg={e}",
                    file=sys.stderr,
                )
                return f"ERROR: {e}"

        # 2×2 runs
        out_i0q0 = _run(base_path, prompt_q0, "I0q0")
        out_i0q1 = _run(base_path, prompt_q1, "I0q1")
        out_i1q0 = _run(flip_path, prompt_q0, "I1q0")
        out_i1q1 = _run(flip_path, prompt_q1, "I1q1")

        df_out.at[idx, col_i0q0_json] = out_i0q0
        df_out.at[idx, col_i0q1_json] = out_i0q1
        df_out.at[idx, col_i1q0_json] = out_i1q0
        df_out.at[idx, col_i1q1_json] = out_i1q1

        processed_rows += 1

    print(
        f"[INFO] Core eval complete: processed_rows={processed_rows} "
        f"skipped_rows={skipped_rows} skipped_base={skipped_base} skipped_flip={skipped_flip}",
        file=sys.stderr,
    )

    return df_out


