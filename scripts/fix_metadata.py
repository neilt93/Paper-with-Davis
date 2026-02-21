#!/usr/bin/env python3
"""Use GPT-4o-mini (text-only) to fix base_setup, flip_change, indoor_outdoor.

Infers all three fields from base_question + flip_question alone (no images).
Does NOT touch ID, Status, base_question, flip_question, pic_base, pic_flip.
"""

import argparse
import csv
import json
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "Sheets" / "results.csv"
FIELDNAMES = ['ID', 'Status', 'base_setup', 'base_question', 'flip_question',
              'flip_change', 'pic_base', 'pic_flip', 'indoor_outdoor']

load_dotenv(BASE_DIR / ".env")

client = OpenAI()

SYSTEM_PROMPT = """You are helping fix metadata for a vision dataset. Each row has:
- A base_question (asks whether something is visible/readable/present in the BASE image — answer should be NOT_VISIBLE)
- A flip_question (asks the opposite — whether it's NOT visible/readable in the BASE image)

Your job: given the two questions, infer exactly three fields:

1. base_setup: A brief (5-15 word) description of the scene that explains WHY the target thing is not visible. Infer from what the questions are asking about. Example patterns:
   - "Far facade, unaided view." (building number too far to read)
   - "Laptop with text on screen." (text too small)
   - "Sunglasses hide eyes." (gaze direction hidden)
   Focus on what makes visibility ambiguous or impossible.

2. flip_change: A brief description of what physical change was made to CREATE the flip image (where the thing BECOMES visible). Examples:
   - "zoom in"
   - "remove sunglasses"
   - "turn on lights"
   - "move the object into frame"
   Infer this from what would need to change for the base_question's answer to flip.

3. indoor_outdoor: Either "indoor" or "outdoor" — infer from what the questions describe.

Return ONLY valid JSON: {"base_setup": "...", "flip_change": "...", "indoor_outdoor": "..."}"""


def row_needs_update(row):
    """Return True if any of the 3 metadata fields are empty."""
    return (not row.get('base_setup', '').strip()
            or not row.get('flip_change', '').strip()
            or not row.get('indoor_outdoor', '').strip())


def process_row(row):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""Row {row['ID']}:
base_question: {row['base_question']}
flip_question: {row['flip_question']}

Return JSON with base_setup, flip_change, indoor_outdoor."""},
        ]
    )

    text = resp.choices[0].message.content.strip()
    if '{' in text:
        text = text[text.index('{'):text.rindex('}') + 1]
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description="Fix metadata fields using GPT-4o-mini")
    parser.add_argument('--only-missing', action='store_true',
                        help='Only process rows where metadata fields are empty')
    args = parser.parse_args()

    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    to_process = [(i, row) for i, row in enumerate(rows)
                  if not args.only_missing or row_needs_update(row)]

    print(f"Total rows: {total}, processing: {len(to_process)}"
          + (f" (skipping {total - len(to_process)} already filled)" if args.only_missing else ""))

    updated = 0
    errors = 0

    for idx, (i, row) in enumerate(to_process):
        try:
            result = process_row(row)
            row['base_setup'] = result['base_setup']
            row['flip_change'] = result['flip_change']
            row['indoor_outdoor'] = result['indoor_outdoor']
            updated += 1
            print(f"  [{idx+1:3d}/{len(to_process)}] {row['ID']:8s} ok  {result['base_setup'][:50]}")
        except Exception as e:
            errors += 1
            print(f"  [{idx+1:3d}/{len(to_process)}] {row['ID']:8s} ERR {e}")

        # Rate limit
        if (idx + 1) % 10 == 0:
            time.sleep(1)

    # Save
    with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in FIELDNAMES})

    print(f"\nDone: {updated} updated, {errors} errors")
    print(f"Saved to {CSV_PATH}")


if __name__ == '__main__':
    main()
