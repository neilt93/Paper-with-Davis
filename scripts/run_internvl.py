#!/usr/bin/env python3
"""Custom InternVL2 inference script with proper API handling.

Uses InternVL2's native chat() method with official image preprocessing.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from visibility_eval_core import (
    load_table, resolve_image_path, normalize_question,
    PROMPT_TEMPLATE, _save_dataframe
)

# ============== InternVL2 Official Image Preprocessing ==============

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    """Build InternVL2 image transform."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find best aspect ratio for dynamic tiling."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    """Dynamic preprocessing with tiling for InternVL2."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    """Load and preprocess image for InternVL2 (official method)."""
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ============== Model Loading ==============

def load_internvl(model_id: str = "OpenGVLab/InternVL2-2B"):
    """Load InternVL2 with proper settings."""
    from transformers import AutoModel, AutoTokenizer

    print(f"[INFO] Loading {model_id}...")

    # Check dtype support
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32  # CPU needs fp32

    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    print(f"[INFO] Model loaded on {next(model.parameters()).device}, dtype={dtype}")
    return model, tokenizer

# ============== Query Function ==============

def run_internvl_query(model, tokenizer, image_path: str, prompt: str) -> str:
    """Run a single query through InternVL2 using its native chat API."""
    try:
        # Load and preprocess image using official method
        pixel_values = load_image(image_path, max_num=6)

        # Move to model device and dtype
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        pixel_values = pixel_values.to(device=device, dtype=dtype)

        # InternVL2 expects <image> token in the prompt
        question = f"<image>\n{prompt}"

        # Use official chat() signature: model.chat(tokenizer, pixel_values, question, generation_config, history=None)
        generation_config = dict(max_new_tokens=128, do_sample=False)
        response = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            history=None,
        )
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"ERROR: {e}"


def extract_json(text: str) -> str:
    """Extract JSON from response text with brace matching."""
    import re

    # Find first { and match to closing }
    start = text.find('{')
    if start == -1:
        return text

    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    json.loads(candidate)
                    return candidate
                except:
                    pass
                break

    # Fallback: simple regex for flat JSON
    matches = list(re.finditer(r'\{[^{}]*\}', text))
    if matches:
        candidate = matches[-1].group()
        try:
            json.loads(candidate)
            return candidate
        except:
            pass
    return text

# ============== Main ==============

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run InternVL2 on visibility dataset")
    parser.add_argument("--input", "-i", required=True, help="Input CSV")
    parser.add_argument("--model", "-m", default="OpenGVLab/InternVL2-2B", help="Model ID")
    parser.add_argument("--out", "-o", default=None, help="Output path")
    parser.add_argument("--only-missing", action="store_true", help="Skip completed rows")
    args = parser.parse_args()

    # Determine output path
    model_tag = args.model.replace("/", "_")
    out_path = args.out or f"{os.path.splitext(args.input)[0]}.{model_tag}.vlm.csv"

    # Load data
    if args.only_missing and os.path.exists(out_path):
        df = load_table(out_path)
        print(f"[INFO] Resuming from {out_path}")
    else:
        df = load_table(args.input)

    # Column names
    col_i0q0 = f"{model_tag}_I0q0_json"
    col_i0q1 = f"{model_tag}_I0q1_json"
    col_i1q0 = f"{model_tag}_I1q0_json"
    col_i1q1 = f"{model_tag}_I1q1_json"

    for col in [col_i0q0, col_i0q1, col_i1q0, col_i1q1]:
        if col not in df.columns:
            df[col] = ""

    # Load model
    model, tokenizer = load_internvl(args.model)

    # Process rows
    done_mask = df["Status"] == "Done"
    done_indices = df[done_mask].index

    for idx in done_indices:
        row = df.loc[idx]
        row_id = str(row.get("ID", row.get("id", idx)))

        # Check if already done
        existing = [str(row.get(c, "")).strip() for c in [col_i0q0, col_i0q1, col_i1q0, col_i1q1]]
        if args.only_missing and all(e and not e.startswith("ERROR") for e in existing):
            print(f"[SKIP] {row_id} already complete")
            continue

        # Get questions
        q0 = normalize_question(str(row.get("base_question", "")))
        q1 = normalize_question(str(row.get("flip_question", "")))

        if not q0 or not q1:
            print(f"[SKIP] {row_id} missing questions")
            continue

        # Get image paths
        base_path = resolve_image_path(str(row.get("pic_base", "")))
        flip_path = resolve_image_path(str(row.get("pic_flip", "")))

        if not base_path or not flip_path:
            print(f"[SKIP] {row_id} missing images")
            continue

        print(f"[RUN] {row_id}")

        # Build prompts
        prompt_q0 = PROMPT_TEMPLATE.format(question=q0)
        prompt_q1 = PROMPT_TEMPLATE.format(question=q1)

        # Run 2x2
        results = {}
        for cell, img, prompt in [
            (col_i0q0, base_path, prompt_q0),
            (col_i0q1, base_path, prompt_q1),
            (col_i1q0, flip_path, prompt_q0),
            (col_i1q1, flip_path, prompt_q1),
        ]:
            resp = run_internvl_query(model, tokenizer, img, prompt)
            results[cell] = extract_json(resp)

        # Save results
        for col, val in results.items():
            df.at[idx, col] = val

        # Incremental save
        _save_dataframe(df, out_path)

    # Final save
    df.to_csv(out_path, index=False)
    print(f"[INFO] Done. Saved to {out_path}")


if __name__ == "__main__":
    main()
