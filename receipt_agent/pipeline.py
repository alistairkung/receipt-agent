"""Integration pipeline for receipt extraction, validation, and calculation."""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from langchain_core.runnables import RunnableLambda
from langchain_deepseek import ChatDeepSeek

from lib.receipt_calculator import ReceiptCalculator
from lib.receipt_extractor import build_receipt_extraction_chain
from lib.receipt_validator import ReceiptValidator


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


@dataclass(frozen=True)
class ReceiptRunResult:
    """Aggregate result for one folder of receipts."""

    amount_paid: Decimal
    amount_without_discounts: Decimal
    receipts_processed: int
    fallback_receipts: tuple[str, ...]


def image_files(folder: Path) -> list[Path]:
    """Return supported image files directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image as a data URL for a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


class ReceiptPipeline:
    """Compose probabilistic receipt extraction with deterministic safeguards."""

    def __init__(
        self,
        raw_chain: Any,
        *,
        validator: ReceiptValidator | None = None,
        calculator: ReceiptCalculator | None = None,
        max_attempts: int = 10,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self.raw_chain = raw_chain
        self.validator = validator or ReceiptValidator()
        self.calculator = calculator or ReceiptCalculator()

        self.validated_chain = (
            self.raw_chain | RunnableLambda(self.validator.validate)
        ).with_retry(
            retry_if_exception_type=(ValueError,),
            stop_after_attempt=max_attempts,
        )

    @classmethod
    def from_deepseek(
        cls,
        api_key: str,
        *,
        max_attempts: int = 10,
    ) -> "ReceiptPipeline":
        """Build the production pipeline around the required DeepSeek vision model."""
        llm = ChatDeepSeek(
            model="deepseek-v4-flash-vision-exp",
            api_key=api_key,
            max_tokens=6000,
            timeout=30,
            max_retries=2,
            extra_body={"thinking": {"type": "disabled"}},
        )

        raw_chain = build_receipt_extraction_chain(llm)
        return cls(raw_chain, max_attempts=max_attempts)

    def process_folder(
        self,
        folder: Path,
        *,
        log: Callable[[str], None] = print,
    ) -> ReceiptRunResult:
        """Process every supported receipt image in a folder."""
        if not folder.is_dir():
            raise ValueError(f"not a folder: {folder}")

        images = image_files(folder)
        if not images:
            raise ValueError(f"no supported images found in {folder}")

        receipts: list[dict[str, Any]] = []
        fallback_receipts: list[str] = []

        for image in images:
            log(f"Processing {image.name}")
            image_input = {"image_url": image_data_url(image)}

            try:
                receipt = self.validated_chain.invoke(image_input)
                log("  correctly validated")
            except ValueError as error:
                log(f"  validation failed after retries: {error}")
                log("  using raw extraction fallback")
                receipt = self.raw_chain.invoke(image_input)
                fallback_receipts.append(image.name)

            receipts.append(receipt)

        answers = self.calculator.calculate(receipts)

        return ReceiptRunResult(
            amount_paid=answers["amount_paid"],
            amount_without_discounts=answers["amount_without_discounts"],
            receipts_processed=len(receipts),
            fallback_receipts=tuple(fallback_receipts),
        )
