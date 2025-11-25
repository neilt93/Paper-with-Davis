"""Push a local CSV (after eval) back to Google Sheets as a new tab.

This is stage 3 of the cloud workflow:
  1) Pull the tab with `scripts/cloud/pull_from_sheets.py`.
  2) Run the local eval script on the CSV.
  3) Use this script to create/overwrite a tab with the results.
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List

import pandas as pd


def get_gspread_client(creds_path: str):
    """
    Create an authenticated gspread client using a service account JSON.
    """
    try:
        import gspread  # type: ignore
        from google.oauth2.service_account import Credentials  # type: ignore
    except ImportError:
        raise RuntimeError(
            "Missing gspread / google-auth. Install with:\n"
            "  pip install gspread google-auth"
        )

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
    client = gspread.authorize(creds)
    return client


def push_csv_to_sheet(spreadsheet_id: str, sheet_name: str, csv_path: str, creds_path: str) -> None:
    """
    Push a local CSV to a (new or existing) tab in the same Google Sheet.
    If the sheet already exists, it will be cleared and overwritten.
    """
    client = get_gspread_client(creds_path)
    sh = client.open_by_key(spreadsheet_id)

    try:
        ws = sh.worksheet(sheet_name)
        ws.clear()
    except Exception:
        # Worksheet doesn't exist; create it
        ws = sh.add_worksheet(title=sheet_name, rows=1, cols=1)

    df = pd.read_csv(csv_path)
    rows: List[List[str]] = [df.columns.tolist()] + df.astype(str).values.tolist()

    # Resize sheet to fit data and update in one call
    ws.resize(rows=len(rows), cols=len(rows[0]))
    ws.update("A1", rows)


def main():
    parser = argparse.ArgumentParser(
        description="Push a local CSV (after eval) to Google Sheets as a new tab."
    )
    parser.add_argument(
        "--spreadsheet-id",
        required=True,
        help="Google Sheets spreadsheet ID (from the URL).",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the local CSV to push (e.g. Sheets/11-14-25.gpt.vlm.csv).",
    )
    parser.add_argument(
        "--sheet-name",
        default=None,
        help=(
            "Name of the output sheet/tab (e.g. '11-14-25.gpt'). "
            "If omitted, derived from the input filename."
        ),
    )
    parser.add_argument(
        "--with-timestamp",
        action="store_true",
        help=(
            "If set, append a YYYY-MM-DD_HHMM timestamp to the sheet name, "
            "e.g. '11-14-25.gpt.2025-11-25_1530'."
        ),
    )
    parser.add_argument(
        "--creds",
        default=None,
        help="Path to Google service account JSON (default: GOOGLE_APPLICATION_CREDENTIALS env var).",
    )
    args = parser.parse_args()

    creds_path = args.creds or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        print(
            "[ERROR] No Google credentials provided. Set GOOGLE_APPLICATION_CREDENTIALS "
            "or pass --creds path/to/service_account.json",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"[ERROR] Input CSV not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Derive sheet name if not provided
    if args.sheet_name:
        base_name = args.sheet_name
    else:
        base_name = os.path.splitext(os.path.basename(args.input))[0]

    if args.with_timestamp:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        sheet_name = f"{base_name}.{stamp}"
    else:
        sheet_name = base_name

    print(
        f"[INFO] Pushing '{args.input}' to spreadsheet {args.spreadsheet_id} "
        f"as sheet '{sheet_name}'",
        file=sys.stderr,
    )

    try:
        push_csv_to_sheet(args.spreadsheet_id, sheet_name, args.input, creds_path)
    except Exception as e:
        print(f"[ERROR] Failed to push results to Google Sheets: {e}", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Done.", file=sys.stderr)


if __name__ == "__main__":
    main()


