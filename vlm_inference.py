import argparse
import os
import sys
from typing import Optional

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


def to_relative(path: str) -> str:
    """
    Take something like:
        /Users/neiltripathi/Documents/Paper with Davis/Images/IMG_0294.jpeg'
    and return:
        ./IMG_0294.jpeg
    """
    if not isinstance(path, str):
        return ""
    cleaned = path.strip().strip("'").strip('"')
    filename = os.path.basename(cleaned)
    if not filename:
        return ""
    return f"./{filename}"


def resolve_image_path(raw_path: str) -> Optional[str]:
    """
    Prefer the original absolute path if it exists; otherwise try common local
    locations based on just the filename, including ./Images/Images.
    """
    if not isinstance(raw_path, str):
        return None

    cleaned = raw_path.strip().strip("'").strip('"')

    # 1) If the path as-is exists (e.g. already-correct relative/absolute), use it
    if cleaned and os.path.exists(cleaned):
        print(f"[INFO] Using image (as-is): {cleaned}", file=sys.stderr)
        return cleaned

    filename = os.path.basename(cleaned)
    if not filename:
        return None

    cwd = os.getcwd()

    # 2) Try likely folders in your workspace
    candidate_dirs = [
        cwd,  # .
        os.path.join(cwd, "Images"),  # ./Images
        os.path.join(cwd, "Images", "Images"),  # ./Images/Images
        os.path.join(cwd, "Images Sheet Versions"),  # extra safety
    ]

    for d in candidate_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            print(f"[INFO] Using image: {candidate}", file=sys.stderr)
            return candidate

    # 3) If nothing matched, warn and return None
    print(f"[WARN] Could not resolve image path for: {raw_path} (filename={filename})", file=sys.stderr)
    return None


def print_markdown_table(df_done: pd.DataFrame) -> None:
    print("| id | vlm_base | vlm_flip |")
    print("| --- | --- | --- |")
    for _, row in df_done.iterrows():
        print(f"| {row['id']} | {row['vlm_base']} | {row['vlm_flip']} |")


def load_model_and_processor(model_id: str):
    from transformers import AutoProcessor, AutoModelForCausalLM
    import torch

    if torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def run_vlm(model, processor, image_path: str, prompt_text: str, max_new_tokens: int = 128) -> str:
    import torch

    image = Image.open(image_path).convert("RGB")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return text.strip()


def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path.lower())[1]
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input file extension: {ext}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate VLM paths and run LLaVA-OneVision-1.5-4B-Instruct over base/flip images."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="11-13-25.csv",
        help="Input CSV/Excel file (default: 11-13-25.csv)",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Output CSV file (default: <input_basename>.vlm.csv)",
    )
    parser.add_argument(
        "--model-id",
        default="lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
        help="Hugging Face model id to use",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = read_table(args.input)

    required_cols = {"id", "Status", "pic_base", "pic_flip", "base_question"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Missing required columns: {', '.join(sorted(missing))}", file=sys.stderr)
        sys.exit(1)

    df_done = df[df["Status"] == "Done"].copy()
    df_done = df_done.sort_values("id")

    # Build relative paths for markdown
    df_done["vlm_base"] = df_done["pic_base"].apply(to_relative)
    df_done["vlm_flip"] = df_done["pic_flip"].apply(to_relative)

    # 1) Print markdown table you can paste into the sheet
    print_markdown_table(df_done)

    # 2) Run VLM on each base/flip image with per-row base_question
    try:
        model, processor = load_model_and_processor(args.model_id)
    except Exception as e:
        print(f"Failed to load model '{args.model_id}': {e}", file=sys.stderr)
        sys.exit(1)

    base_outputs = []
    flip_outputs = []

    for _, row in df_done.iterrows():
        question = row["base_question"]
        q_text = "" if not isinstance(question, str) else question.strip()
        full_prompt = PROMPT_TEMPLATE.format(question=q_text)

        # Base image
        base_path_abs = resolve_image_path(row["pic_base"])
        if base_path_abs is None:
            base_outputs.append("")
        else:
            try:
                resp_base = run_vlm(model, processor, base_path_abs, full_prompt)
            except Exception as e:
                resp_base = f"ERROR: {e}"
            base_outputs.append(resp_base)

        # Flip image
        flip_path_abs = resolve_image_path(row["pic_flip"])
        if flip_path_abs is None:
            flip_outputs.append("")
        else:
            try:
                resp_flip = run_vlm(model, processor, flip_path_abs, full_prompt)
            except Exception as e:
                resp_flip = f"ERROR: {e}"
            flip_outputs.append(resp_flip)

    df_done["vlm_base_json"] = base_outputs
    df_done["vlm_flip_json"] = flip_outputs

    # 3) Write results out
    out_path = args.out
    if out_path is None:
        base, ext = os.path.splitext(args.input)
        out_path = f"{base}.vlm.csv"

    df_done.to_csv(out_path, index=False)
    print(f"\nSaved VLM results to: {out_path}")


if __name__ == "__main__":
    main()


