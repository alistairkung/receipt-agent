from decimal import Decimal

from langchain_core.runnables import RunnableLambda

from receipt_agent.pipeline import ReceiptPipeline


VALID_RECEIPT = {
    "items": [{"description": "Example item", "original_line_amount": 10.00}],
    "discounts": [],
    "subtotal_after_discounts": 10.00,
    "rounding": 0.00,
    "amount_paid_after_rounding": 10.00,
}


def test_pipeline_processes_valid_receipt(tmp_path):
    image = tmp_path / "receipt.jpg"
    image.write_bytes(b"fake-image")

    raw_chain = RunnableLambda(lambda _: VALID_RECEIPT)
    pipeline = ReceiptPipeline(raw_chain, max_attempts=1)

    result = pipeline.process_folder(tmp_path, log=lambda _: None)

    assert result.receipts_processed == 1
    assert result.fallback_receipts == ()
    assert result.amount_paid == Decimal("10.0")
    assert result.amount_without_discounts == Decimal("10.0")


def test_pipeline_falls_back_to_raw_extraction_after_validation_failure(tmp_path):
    image = tmp_path / "receipt.jpg"
    image.write_bytes(b"fake-image")

    invalid_but_calculable_receipt = {
        "items": [{"description": "Example item", "original_line_amount": 10.00}],
        "discounts": [],
        "subtotal_after_discounts": 9.00,
        "rounding": 0.00,
        "amount_paid_after_rounding": 9.00,
    }

    raw_chain = RunnableLambda(lambda _: invalid_but_calculable_receipt)
    pipeline = ReceiptPipeline(raw_chain, max_attempts=1)

    result = pipeline.process_folder(tmp_path, log=lambda _: None)

    assert result.fallback_receipts == ("receipt.jpg",)
    assert result.amount_paid == Decimal("9.0")
    assert result.amount_without_discounts == Decimal("10.0")
