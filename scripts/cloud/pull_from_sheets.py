"""Pull a Google Sheets tab into a local CSV for visibility evaluation.

This is stage 1 of the cloud workflow:
  1) Use this script to pull a tab into `Sheets/<name>.csv`.
  2) Run the local eval script on that CSV.
  3) Optionally push results back with `scripts/cloud/push_to_sheets.py`.

You can either name the sheet explicitly (`--input-sheet`) or ask for
the most recent sheet whose title starts with a prefix (`--latest-prefix`).
"""

import argparse
import os
import sys
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


def pull_sheet_to_csv(spreadsheet_id: str, sheet_name: str, out_csv: str, creds_path: str) -> None:
    """
    Pull a Google Sheet tab into a local CSV file.
    """
    client = get_gspread_client(creds_path)
    sh = client.open_by_key(spreadsheet_id)
    ws = sh.worksheet(sheet_name)

    values: List[List[str]] = ws.get_all_values()
    if not values:
        raise RuntimeError(f"Sheet '{sheet_name}' is empty.")

    header = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Pull a Google Sheet tab into a local CSV for visibility eval."
    )
    parser.add_argument(
        "--spreadsheet-id",
        required=True,
        help="Google Sheets spreadsheet ID (from the URL).",
    )
    parser.add_argument(
        "--input-sheet",
        help="Name of the input sheet/tab to pull (e.g. '11-14-25').",
    )
    parser.add_argument(
        "--latest-prefix",
        default=None,
        help=(
            "If set, ignore --input-sheet and pull the sheet whose title starts with "
            "this prefix and is lexicographically last (e.g. latest timestamp)."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: Sheets/<input-sheet>.csv).",
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

    # Resolve which sheet to pull
    if args.latest_prefix:
        # Discover the most recent sheet whose name starts with the prefix
        client = get_gspread_client(creds_path)
        sh = client.open_by_key(args.spreadsheet_id)
        candidates = [ws.title for ws in sh.worksheets() if ws.title.startswith(args.latest_prefix)]
        if not candidates:
            print(
                f"[ERROR] No sheets found with prefix '{args.latest_prefix}' in spreadsheet {args.spreadsheet_id}",
                file=sys.stderr,
            )
            sys.exit(1)
        # Lexicographically last title (works well with YYYY-MM-DD_HHMM suffixes)
        sheet_name = sorted(candidates)[-1]
        print(f"[INFO] Using latest sheet with prefix '{args.latest_prefix}': '{sheet_name}'", file=sys.stderr)
    else:
        if not args.input_sheet:
            print("[ERROR] Either --input-sheet or --latest-prefix must be provided.", file=sys.stderr)
            sys.exit(1)
        sheet_name = args.input_sheet

    out_path = args.out
    if out_path is None:
        os.makedirs("Sheets", exist_ok=True)
        out_path = os.path.join("Sheets", f"{sheet_name}.csv")

    print(
        f"[INFO] Pulling sheet '{sheet_name}' from spreadsheet {args.spreadsheet_id} -> {out_path}",
        file=sys.stderr,
    )
    try:
        pull_sheet_to_csv(args.spreadsheet_id, sheet_name, out_path, creds_path)
    except Exception as e:
        print(f"[ERROR] Failed to pull sheet: {e}", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Done.", file=sys.stderr)


if __name__ == "__main__":
    main()


