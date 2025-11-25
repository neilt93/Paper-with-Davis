import argparse
import os
import sys
import subprocess
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


def pull_sheet_to_csv(spreadsheet_id: str, sheet_name: str, tmp_csv: str, creds_path: str) -> None:
    """
    Pull a Google Sheet tab into a local CSV.
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
    df.to_csv(tmp_csv, index=False)


def push_csv_to_new_sheet(spreadsheet_id: str, sheet_name: str, csv_path: str, creds_path: str) -> None:
    """
    Push a local CSV to a new tab in the same Google Sheet.
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
        description="Pull a Google Sheet tab, run run_visibility_eval.py locally, and push results back as a new tab."
    )
    parser.add_argument(
        "--spreadsheet-id",
        required=True,
        help="Google Sheets spreadsheet ID (from the URL).",
    )
    parser.add_argument(
        "--input-sheet",
        required=True,
        help="Name of the input sheet/tab to pull (e.g. '11-14-25').",
    )
    parser.add_argument(
        "--output-sheet",
        required=True,
        help="Name of the output sheet/tab to create/update with results (e.g. '11-14-25.gpt').",
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name to pass to run_visibility_eval.py (e.g. gpt, gemini, llava-onevision).",
    )
    parser.add_argument(
        "--creds",
        default=None,
        help="Path to Google service account JSON (default: GOOGLE_APPLICATION_CREDENTIALS env var).",
    )
    parser.add_argument(
        "--tmp-dir",
        default=".tmp_sheets_eval",
        help="Temporary directory for intermediate CSVs (default: .tmp_sheets_eval).",
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

    os.makedirs(args.tmp_dir, exist_ok=True)
    tmp_in = os.path.join(args.tmp_dir, f"{args.input_sheet}.in.csv")
    tmp_out = os.path.join(args.tmp_dir, f"{args.input_sheet}.{args.model}.out.csv")

    print(
        f"[INFO] Pulling sheet '{args.input_sheet}' from spreadsheet {args.spreadsheet_id} -> {tmp_in}",
        file=sys.stderr,
    )
    try:
        pull_sheet_to_csv(args.spreadsheet_id, args.input_sheet, tmp_in, creds_path)
    except Exception as e:
        print(f"[ERROR] Failed to pull sheet: {e}", file=sys.stderr)
        sys.exit(1)

    # Run local visibility eval script on the pulled CSV
    cmd = [
        sys.executable,
        "run_visibility_eval.py",
        "--input",
        tmp_in,
        "--model",
        args.model,
        "--out",
        tmp_out,
    ]
    print(f"[INFO] Running: {' '.join(cmd)}", file=sys.stderr)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] run_visibility_eval.py failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"[INFO] Pushing results from {tmp_out} to new sheet '{args.output_sheet}'",
        file=sys.stderr,
    )
    try:
        push_csv_to_new_sheet(args.spreadsheet_id, args.output_sheet, tmp_out, creds_path)
    except Exception as e:
        print(f"[ERROR] Failed to push results to Google Sheets: {e}", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Done.", file=sys.stderr)


if __name__ == "__main__":
    main()


