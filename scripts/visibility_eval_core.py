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
    OpenAI GPT vision (e.g. gpt-4.1 / gpt-5.2-pro).
    Requires OPENAI_API_KEY in environment.
    """

    def __init__(self, model: str = "gpt-5.2-pro"):
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
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ]
        resp = self.client.chat.completions.create(model=self.model, messages=msg, max_tokens=1024)
        text = resp.choices[0].message.content
        return ModelResponse(raw_text=text)


class GeminiAdapter(BaseAdapter):
    """
    Google Gemini 1.5 vision via new google-genai client (v1).
    Falls back to legacy google-generativeai if google-genai is unavailable.
    Requires GEMINI_API_KEY in environment.
    """

    def __init__(self, model: str = "gemini-3-pro-preview"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment.")
        # Prefer new google-genai client; fall back to google-generativeai if not available
        self._use_new_client = False
        self._model_name = model
        try:
            from google import genai  # type: ignore
            # Defer import of types to avoid hard dependency when falling back
            from google.genai import types as genai_types  # type: ignore
            self._use_new_client = True
            self._genai = genai
            self._genai_types = genai_types
            self._client = genai.Client(api_key=api_key)
        except Exception:
            # Legacy fallback (deprecated by Google, still works in some environments)
            try:
                import google.generativeai as genai_legacy  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Missing google-genai (preferred) or google-generativeai (legacy). "
                    "Install with `pip install google-genai` (preferred) or `pip install google-generativeai`."
                ) from e
            genai_legacy.configure(api_key=api_key)
            self._legacy_model = genai_legacy.GenerativeModel(model)
            self._genai_legacy = genai_legacy

    def answer(self, image_path: str, prompt: str) -> ModelResponse:
        # New client path (v1 Responses API)
        if self._use_new_client:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            # Try to guess mime type from extension; default to JPEG
            import mimetypes
            mime, _ = mimetypes.guess_type(image_path)
            mime = mime or "image/jpeg"
            try:
                part_image = self._genai_types.Part.from_bytes(data=img_bytes, mime_type=mime)  # type: ignore[attr-defined]
            except Exception:
                # Fallback: older versions expose Part under google.genai.types directly
                part_image = self._genai.types.Part.from_bytes(data=img_bytes, mime_type=mime)  # type: ignore[attr-defined]
            resp = self._client.models.generate_content(  # type: ignore[attr-defined]
                model=self._model_name,
                contents=[part_image, prompt],
            )
            # Robust extraction across library versions
            text = getattr(resp, "text", None) or getattr(resp, "output_text", None)
            if not text:
                try:
                    # Try to drill into candidates if needed
                    cand = resp.candidates[0]
                    # Prefer aggregated text if available
                    text = getattr(cand, "text", None)
                    if not text and getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                        parts = cand.content.parts
                        # Concatenate any text parts
                        text = "".join(getattr(p, "text", "") for p in parts if hasattr(p, "text"))
                except Exception:
                    text = ""
            return ModelResponse(raw_text=(text or "").strip())

        # Legacy client path (v1beta generateContent)
        img = Image.open(image_path)
        resp = self._legacy_model.generate_content([prompt, img])  # type: ignore[attr-defined]
        return ModelResponse(raw_text=getattr(resp, "text", "") or "")


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
                AutoModel,
                AutoConfig,
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
        self.AutoModel = AutoModel
        self.AutoConfig = AutoConfig
        self.AutoModelForImageTextToText = AutoModelForImageTextToText  # type: ignore[attr-defined]

        if torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Some community quantized repos have incomplete/misconfigured image processor metadata.
        # If AutoProcessor can't be inferred, fall back to the base Qwen2.5-VL processor.
        is_molmo = "molmo" in model_id.lower()
        try:
            processor_kwargs = {
                "trust_remote_code": True,
            }
            if is_molmo:
                processor_kwargs["use_fast"] = False  # Silences the warning for Molmo2
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                **processor_kwargs,
            )
        except Exception as e:
            msg = str(e)
            if "Unrecognized image processor" in msg or "image_processor_type" in msg:
                self.processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    trust_remote_code=True,
                )
            else:
                raise

        # Special-case Qwen2.5-VL, Molmo2, PaliGemma2, InternVL2, LLaVA, SmolVLM, and Llama Vision to avoid the AutoModelForCausalLM config error
        is_qwen_vision = "qwen2.5-vl" in model_id.lower() or "qwen2_5_vl" in model_id.lower()
        is_paligemma = "paligemma" in model_id.lower()
        is_llama_vision = "llama" in model_id.lower() and "vision" in model_id.lower()
        is_internvl = "internvl" in model_id.lower()
        is_llava = "llava" in model_id.lower()
        is_smolvlm = "smolvlm" in model_id.lower()
        is_phi_vision = "phi" in model_id.lower() and "vision" in model_id.lower()

        # Store model type for conditional behavior in answer()
        self._is_llama_vision = is_llama_vision

        # Check if we should use 4-bit quantization for Llama (to reduce memory)
        use_4bit_quantization = False
        if is_llama_vision:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
                import bitsandbytes  # type: ignore
                use_4bit_quantization = True
            except ImportError:
                pass  # Will use full precision

        if is_qwen_vision and self.AutoModelForImageTextToText is not None:  # type: ignore[truthy-function]
            ModelCls = self.AutoModelForImageTextToText
            self.model = ModelCls.from_pretrained(
                model_id,
                dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        elif is_molmo:
            if self.AutoModelForImageTextToText is None:  # type: ignore[comparison-overlap]
                raise RuntimeError(
                    "Molmo2 requires AutoModelForImageTextToText which is not available. "
                    "Please upgrade transformers: pip install --upgrade transformers"
                )
            # Molmo2 uses AutoModelForImageTextToText with trust_remote_code
            self.model = self.AutoModelForImageTextToText.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map="auto",
                dtype=dtype,
            )
        elif is_paligemma:
            # PaliGemma2 uses AutoModelForImageTextToText with trust_remote_code
            if self.AutoModelForImageTextToText is None:  # type: ignore[comparison-overlap]
                raise RuntimeError(
                    "PaliGemma2 requires AutoModelForImageTextToText which is not available. "
                    "Please upgrade transformers: pip install --upgrade transformers"
                )
            self.model = self.AutoModelForImageTextToText.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map="auto",
                dtype=dtype,
            )
        elif is_smolvlm:
            # SmolVLM uses AutoModelForImageTextToText (Idefics3-based)
            from transformers import AutoModelForImageTextToText
            # Check bfloat16 support - fall back to float16 on older GPUs
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                smol_dtype = torch.bfloat16
            else:
                smol_dtype = dtype
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=smol_dtype,
                device_map="auto",
                _attn_implementation="eager",
            )
        elif is_phi_vision:
            # Phi-3.5-Vision uses AutoModelForCausalLM with trust_remote_code
            # Check bfloat16 support - fall back to float16 on older GPUs
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                phi_dtype = torch.bfloat16
            else:
                phi_dtype = dtype
            self.model = self.AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True,
                dtype=phi_dtype,
                _attn_implementation="eager",
            )
        elif is_llava:
            # LLaVA-OneVision models use custom architecture, need AutoModel with trust_remote_code
            self.model = self.AutoModel.from_pretrained(
                model_id,
                dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        elif is_internvl:
            # InternVL2 needs exact loading pattern from HuggingFace docs
            # Use bfloat16 if supported, otherwise fall back to float16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                internvl_dtype = torch.bfloat16
            else:
                internvl_dtype = dtype
            self.model = self.AutoModel.from_pretrained(
                model_id,
                dtype=internvl_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",  # Use device_map instead of hard-coded .cuda()
            )
        elif is_llama_vision and use_4bit_quantization:
            # Load Llama Vision with 4-bit quantization to reduce memory
            from transformers import BitsAndBytesConfig  # type: ignore
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            ModelCls = self.AutoModelForCausalLM
            self.model = ModelCls.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        else:
            ModelCls = self.AutoModelForCausalLM
            self.model = ModelCls.from_pretrained(
                model_id,
                dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        # Get the actual device from the model's parameters (skip meta tensors).
        # This avoids "Cannot copy out of meta tensor" errors with quantized models.
        self.device = None
        try:
            # Iterate through parameters and find the first non-meta tensor
            for param in self.model.parameters():
                try:
                    param_device = param.device
                    if param_device.type != "meta":
                        self.device = param_device
                        break
                except RuntimeError:
                    # Skip meta tensors
                    continue
            
            # If we didn't find a non-meta parameter, try named_parameters
            if self.device is None:
                for name, param in self.model.named_parameters():
                    try:
                        param_device = param.device
                        if param_device.type != "meta":
                            self.device = param_device
                            break
                    except RuntimeError:
                        continue
            
            # Final fallback: assume CUDA if available
            if self.device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            # Fallback if anything goes wrong
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        # Try to use chat template, but fall back to direct prompt for models without templates (e.g., PaliGemma2)
        try:
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
            inputs = self.processor(text=tpl, images=image, return_tensors="pt").to(self.device)
        except (ValueError, TypeError, AttributeError) as e:
            # If chat template is not available, use the prompt directly
            # For PaliGemma2 and similar models, add <image> token at the beginning
            if "chat template" in str(e).lower() or "apply_chat_template" in str(e).lower():
                # Check if this is a PaliGemma processor that needs <image> tokens
                processor_type = type(self.processor).__name__
                if "PaliGemma" in processor_type or "paligemma" in str(type(self.processor)).lower():
                    # PaliGemma expects <image> token at the beginning of the text
                    # Also add explicit JSON instruction at the end to force JSON output
                    prompt_with_image = f"<image>{prompt}\n\nOutput your response as JSON only, starting with {{ and ending with }}."
                    inputs = self.processor(text=prompt_with_image, images=image, return_tensors="pt").to(self.device)
                else:
                    inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            else:
                raise
        
        # Filter out processor keys that aren't used by model.generate()
        # ONLY for Llama 3.2 Vision - its processor returns keys like pixel_values, aspect_ratio_ids, aspect_ratio_mask
        # that should not be passed to generate() - these are for forward() only
        # Other VLMs (InternVL2, Qwen, LLaVA, etc.) DO need pixel_values passed to generate()
        if getattr(self, '_is_llama_vision', False):
            keys_to_remove = {"pixel_values", "aspect_ratio_ids", "aspect_ratio_mask"}
            inputs_for_generate = {k: v for k, v in inputs.items() if k not in keys_to_remove}
        else:
            inputs_for_generate = inputs

        with self.torch.no_grad():
            ids = self.model.generate(**inputs_for_generate, max_new_tokens=128)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return ModelResponse(raw_text=text.strip())


class ClaudeAdapter(BaseAdapter):
    """
    Anthropic Claude (vision-capable).
    Requires ANTHROPIC_API_KEY in environment and anthropic installed.
    """

    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in environment.")
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError:
            raise RuntimeError("Missing anthropic. Install with `pip install anthropic`.")
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def answer(self, image_path: str, prompt: str) -> ModelResponse:
        # Build Claude messages with image + text.
        # Compress/resize to stay under Anthropic's 5MB per-image limit.
        import io
        from PIL import Image as PILImage  # avoid name clash
        img = PILImage.open(image_path).convert("RGB")
        # Downscale if very large
        max_side = 1536
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
        # Compress iteratively until <= 5MB
        buf = io.BytesIO()
        quality = 85
        while True:
            buf.seek(0)
            buf.truncate(0)
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            size = buf.tell()
            if size <= 5 * 1024 * 1024 or quality <= 60:
                break
            quality -= 5
        img_bytes = buf.getvalue()
        b64 = __import__("base64").b64encode(img_bytes).decode("utf-8")
        mime = "image/jpeg"

        msg = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system="You must respond with ONLY a valid JSON object. No prose, no explanations, no markdown. Just the JSON.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": mime, "data": b64},
                        },
                        {"type": "text", "text": prompt + "\n\nIMPORTANT: Respond with ONLY the JSON object, no other text."},
                    ],
                }
            ],
        )
        # Extract text from content blocks
        text_parts = []
        for part in getattr(msg, "content", []) or []:
            if getattr(part, "type", "") == "text":
                text_parts.append(getattr(part, "text", "") or "")
        text = "".join(text_parts).strip()
        return ModelResponse(raw_text=text)

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
                AutoConfig,
            )
            try:
                from transformers import AutoModelForImageTextToText  # type: ignore
            except ImportError:
                AutoModelForImageTextToText = None  # type: ignore[assignment]
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
            except ImportError:
                BitsAndBytesConfig = None  # type: ignore[assignment]
            # Also check if bitsandbytes module itself is importable
            try:
                import bitsandbytes  # type: ignore
                has_bitsandbytes = True
            except (ImportError, ModuleNotFoundError):
                has_bitsandbytes = False
            import torch  # type: ignore
        except ImportError:
            raise RuntimeError("Missing transformers/torch. Install with `pip install transformers torch`.")

        self.torch = torch
        self.AutoProcessor = AutoProcessor
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoConfig = AutoConfig
        self.AutoModelForImageTextToText = AutoModelForImageTextToText  # type: ignore[attr-defined]
        self.BitsAndBytesConfig = BitsAndBytesConfig  # type: ignore[attr-defined]
        self.has_bitsandbytes = has_bitsandbytes

        # Prefer GPU if available. Use FP16 on CUDA for a big speedup.
        # (FP32 is much slower and rarely necessary for inference here.)
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
            # Allow TF32 for additional speed on Ampere+ GPUs.
            try:  # pragma: no cover
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
        else:
            device = "cpu"
            dtype = torch.float32

        # Some quantized community checkpoints (e.g. jarvisvasu/Qwen2.5-VL-3B-Instruct-4bit)
        # have incomplete or older preprocessor configs. In those cases, re-use the official
        # Qwen processor while loading weights from the quantized repo to avoid
        # "Unrecognized image processor" errors.
        processor_id = model_id
        lower_id = model_id.lower()
        if "jarvisvasu/qwen2.5-vl-3b-instruct-4bit" in lower_id:
            processor_id = "Qwen/Qwen2.5-VL-3B-Instruct"

        self.processor = AutoProcessor.from_pretrained(
            processor_id,
            trust_remote_code=True,
        )

        is_qwen_vision = "qwen2.5-vl" in model_id.lower() or "qwen2_5_vl" in model_id.lower()
        if is_qwen_vision and self.AutoModelForImageTextToText is not None:  # type: ignore[truthy-function]
            ModelCls = self.AutoModelForImageTextToText
        else:
            ModelCls = self.AutoModelForCausalLM

        # If the checkpoint declares bitsandbytes 4-bit quantization in its config,
        # prefer loading it in 4-bit (much faster / smaller) instead of full-precision.
        # Example: jarvisvasu/Qwen2.5-VL-3B-Instruct-4bit
        cfg = self.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        quant_cfg = getattr(cfg, "quantization_config", None)
        is_bnb_4bit = False
        if isinstance(quant_cfg, dict):
            is_bnb_4bit = (quant_cfg.get("quant_method") == "bitsandbytes") and bool(quant_cfg.get("load_in_4bit"))

        if is_bnb_4bit:
            # bitsandbytes 4-bit quantization path (e.g. jarvisvasu/Qwen2.5-VL-3B-Instruct-4bit).
            # The checkpoint already carries its own quantization_config, so we let
            # `from_pretrained` honor that and just request device_map="auto".
            if device != "cuda":
                raise RuntimeError("bitsandbytes 4-bit quantization requires CUDA. Use a GPU-enabled environment.")
            if not self.has_bitsandbytes or self.BitsAndBytesConfig is None:
                # bitsandbytes not available - fall back to loading base model or loading without quantization
                print(
                    "[WARN] bitsandbytes not available. Loading base Qwen2.5-VL model instead of quantized version.",
                    file=sys.stderr,
                )
                # Load the base model instead
                base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
                self.model = ModelCls.from_pretrained(
                    base_model_id,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    trust_remote_code=True,
                )
                # When using low_cpu_mem_usage / accelerate dispatch, do NOT call .to(...).
                # Just keep an input device handle.
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                # bitsandbytes is available, try to load the quantized model.
                # However, some pre-quantized checkpoints (e.g. jarvisvasu/Qwen2.5-VL-3B-Instruct-4bit)
                # trigger a meta-tensor dispatch bug on Windows/Python 3.13 when transformers/accelerate
                # tries to dispatch the model. If this fails, fall back to the base FP16 model.
                try:
                    self.model = ModelCls.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        device_map={"": 0},  # Explicit CUDA placement to avoid meta tensor dispatch
                    )
                    self.device = torch.device("cuda")
                except Exception as e:
                    # If loading the quantized checkpoint fails due to meta tensor dispatch issues
                    # (common on Windows), fall back to the base FP16 model for compatibility.
                    if "meta tensor" in str(e).lower() or "to_empty" in str(e).lower() or "Cannot copy out of meta" in str(e):
                        print(
                            f"[WARN] Pre-quantized checkpoint '{model_id}' failed to load (Windows/meta-tensor incompatibility).",
                            file=sys.stderr,
                        )
                        print(
                            f"[WARN] Falling back to base FP16 model 'Qwen/Qwen2.5-VL-3B-Instruct' for compatibility.",
                            file=sys.stderr,
                        )
                        # Load the base model in FP16 instead (still fast, just not quantized)
                        # Use bfloat16 if available (more stable than float16 on some hardware)
                        base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
                        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                            fallback_dtype = torch.bfloat16
                        else:
                            fallback_dtype = dtype
                        self.model = ModelCls.from_pretrained(
                            base_model_id,
                            dtype=fallback_dtype,
                            low_cpu_mem_usage=True,
                            device_map="auto",
                            trust_remote_code=True,
                        )
                        # Get device from model parameters (skip meta tensors)
                        self.device = None
                        try:
                            for param in self.model.parameters():
                                try:
                                    param_device = param.device
                                    if param_device.type != "meta":
                                        self.device = param_device
                                        break
                                except RuntimeError:
                                    continue
                            if self.device is None:
                                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        except Exception:
                            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    else:
                        raise
        else:
            # transformers >= 4.46 prefers `dtype=` over deprecated `torch_dtype=`
            self.model = ModelCls.from_pretrained(
                model_id,
                dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
                trust_remote_code=True,
            )
            # When using low_cpu_mem_usage / accelerate dispatch, do NOT call .to(...).
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def answer(self, image_path: str, prompt: str) -> ModelResponse:
        """
        Run a single image+prompt through Qwen2.5-VL and return a cleaned
        JSON string if possible.
        """
        image = Image.open(image_path).convert("RGB")

        # Downscale for speed; Qwen2.5-VL is expensive at high resolutions.
        # 512 is a good default tradeoff for this dataset.
        max_side = 512
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
        inputs = self.processor(text=tpl, images=image, return_tensors="pt")
        
        # Ensure inputs are on the correct device by checking where model parameters actually are
        # (device_map="auto" can spread model across devices, but inputs usually go to the main device)
        try:
            # Try to get device from model's first parameter
            model_device = next(self.model.parameters()).device
        except (StopIteration, RuntimeError):
            # Fallback to self.device if we can't determine model device
            model_device = self.device
        
        # Move all input tensors to the model's device (handle dict of tensors correctly)
        inputs = {k: v.to(model_device) if isinstance(v, self.torch.Tensor) else v for k, v in inputs.items()}
        
        with self.torch.no_grad():
            # 64 tokens is plenty for a single minified JSON object.
            # Add error handling for CUDA device-side asserts
            try:
                ids = self.model.generate(**inputs, max_new_tokens=64)
            except RuntimeError as e:
                error_str = str(e)
                if "CUDA error" in error_str or "device-side assert" in error_str:
                    # CUDA device-side assert - clear cache and provide better error message
                    if self.torch.cuda.is_available():
                        self.torch.cuda.empty_cache()
                    # This often indicates an incompatibility with the FP16 model or input processing
                    raise RuntimeError(
                        f"CUDA device-side assert during Qwen generation. "
                        f"This may indicate a compatibility issue with FP16 Qwen2.5-VL on Windows. "
                        f"Try using a different model (e.g., LLaVA) or running on CPU. "
                        f"Original error: {error_str}"
                    ) from e
                raise
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]

        # Post-process to keep only the final JSON object if we can find it.
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

    # === FLAGSHIP MODELS ===
    if name in {"gpt5", "gpt-5", "gpt-5.2"}:
        return GPTAdapter(model="gpt-4o")

    if name in {"gemini3", "gemini-3", "gemini-3-pro"}:
        return GeminiAdapter(model="gemini-3-pro-preview")

    if name in {"opus", "claude-opus", "opus-4.5"}:
        return ClaudeAdapter(model="claude-opus-4-5-20251101")

    # === LEGACY MODELS ===
    if name in {"gpt", "gpt4", "openai"}:
        return GPTAdapter(model="gpt-4o")

    if name in {"gemini", "gemini-1.5"}:
        return GeminiAdapter(model="gemini-pro-latest")

    if name in {"claude", "anthropic", "claude-sonnet", "claude-3-5", "claude-3-7"}:
        return ClaudeAdapter(model="claude-3-7-sonnet-20250219")

    # Qwen gets a custom adapter so we can post-process its outputs.
    if name == "qwen-vl":
        return QwenLocalAdapter("Qwen/Qwen2.5-VL-3B-Instruct")
    if name in {"qwen-vl-7b", "qwen-7b"}:
        return QwenLocalAdapter("Qwen/Qwen2.5-VL-7B-Instruct")
    # Quantized (bitsandbytes NF4) Qwen. Works on Windows if CUDA + bitsandbytes are installed.
    if name in {"qwen-vl-4bit", "qwen-vl-bnb4"}:
        return QwenLocalAdapter("jarvisvasu/Qwen2.5-VL-3B-Instruct-4bit")
    # Quantized Qwen (faster). Requires the checkpoint to be loadable in your environment.
    if name in {"qwen-vl-awq", "qwen-vl-3b-awq"}:
        return QwenLocalAdapter("Qwen/Qwen2.5-VL-3B-Instruct-AWQ")

    hf_map = {
        "llava-onevision": "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
        "llava-7b": "lmms-lab/llava-onevision-qwen2-7b-ov",
        "idefics2": "HuggingFaceM4/idefics2-8b",
        # Baselines (may be gated on HF)
        "molmo": "allenai/Molmo2-4B",
        "gemma": "google/paligemma2-3b-mix-448",
        "llama-vision": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        # Mid-tier open source (RTX 3090 compatible)
        # InternVL2-8B: use original repo (requires patch_internvl.py after download)
        "internvl2-8b": "OpenGVLab/InternVL2-8B",
        # InternVL3: newer version with native HF transformers support (no patch needed)
        # See: https://huggingface.co/OpenGVLab/InternVL3-8B-hf
        "internvl3-8b": "OpenGVLab/InternVL3-8B-hf",
        # Tested working with standard transformers
        "smolvlm": "HuggingFaceTB/SmolVLM-Instruct",
        "phi3-vision": "microsoft/Phi-3.5-vision-instruct",
    }
    if name in hf_map:
        return LocalHFAdapter(hf_map[name])

    return LocalHFAdapter(model_name)


def _save_dataframe(df: pd.DataFrame, save_path: Optional[str]) -> None:
    """Helper to save dataframe to CSV or Excel based on extension."""
    if not save_path:
        return
    try:
        out_ext = os.path.splitext(save_path.lower())[1]
        if out_ext in {".xlsx", ".xls"}:
            df.to_excel(save_path, index=False)
        else:
            df.to_csv(save_path, index=False)
        print(f"[SAVE] Incremental save: {save_path}", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Failed to save incrementally to '{save_path}': {e}", file=sys.stderr)


def run_on_dataframe(df: pd.DataFrame, model_name: str, only_missing: bool = False, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Core evaluation logic:
      - verify that required "base" columns are present
      - filter to Status == 'Done' for running the model
      - normalize questions
      - resolve image paths
      - call the chosen model adapter
      - fill per-model 2×2 columns:
          <model>_I0q0_json / <model>_I0q1_json / <model>_I1q0_json / <model>_I1q1_json

    If save_path is provided, saves incrementally after each row to preserve
    progress in case of errors (e.g., 429 rate limits).

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

        # Get existing values, handling NaN properly (pandas reads empty CSV cells as NaN)
        # Also detect error messages (starting with "ERROR:") and prose responses (no valid JSON) and treat them as incomplete
        def _get_value(col_name: str) -> str:
            val = row.get(col_name, "")
            if pd.isna(val) or val == "" or val is None:
                return ""
            val_str = str(val).strip()
            # Treat error messages as incomplete (need retry)
            if val_str.startswith("ERROR:"):
                return ""
            # Check if it's valid JSON with the expected structure
            try:
                import json
                parsed = json.loads(val_str)
                if isinstance(parsed, dict) and "label" in parsed:
                    return val_str  # Valid JSON with label field
            except:
                pass
            # If it's not valid JSON or doesn't have label, and it's long enough to be prose, treat as incomplete
            if len(val_str) > 50 and not val_str.startswith("{"):
                return ""  # Likely prose response without JSON, needs retry
            return val_str

        existing_base = _get_value(col_i0q0_json)
        existing_flip = _get_value(col_i1q0_json)
        existing_i0q1 = _get_value(col_i0q1_json)
        existing_i1q1 = _get_value(col_i1q1_json)

        if only_missing and existing_base and existing_flip and existing_i0q1 and existing_i1q1:
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

        # Save incrementally after each row to preserve progress on errors
        if save_path:
            _save_dataframe(df_out, save_path)

    print(
        f"[INFO] Core eval complete: processed_rows={processed_rows} "
        f"skipped_rows={skipped_rows} skipped_base={skipped_base} skipped_flip={skipped_flip}",
        file=sys.stderr,
    )

    return df_out


