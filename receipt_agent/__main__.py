"""Command-line interface for the standalone receipt agent."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from .pipeline import ReceiptPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract, validate, and total a folder of receipt images."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="folder containing receipt images",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=10,
        help="maximum validated extraction attempts per receipt (default: 10)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set (add it to .env or the environment)")

    pipeline = ReceiptPipeline.from_deepseek(
        api_key,
        max_attempts=args.attempts,
    )

    try:
        result = pipeline.process_folder(args.folder)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print()
    print(f"Processed {result.receipts_processed} receipt(s)")
    print(f"Amount paid: HK${result.amount_paid:.2f}")
    print(f"Without discounts: HK${result.amount_without_discounts:.2f}")

    if result.fallback_receipts:
        print("Raw fallback used for: " + ", ".join(result.fallback_receipts))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
