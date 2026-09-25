"""Validation and normalisation boundary for model-extracted receipts."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from typing import Any


class ReceiptValidator:
    """Validate an untrusted extracted receipt and return a trusted equivalent.

    Intended responsibilities:
    - required receipt/item/discount fields exist;
    - required monetary values are present;
    - monetary values are normalised to Decimal;
    - signs satisfy the receipt-domain contract;
    - extracted totals reconcile.

    Invalid model output should fail explicitly rather than being silently repaired.
    """

    AGGREGATE_AMOUNT_FIELDS = [
        "subtotal_after_discounts",
        "rounding",
        "amount_paid_after_rounding",
    ]

    REQUIRED_FIELDS = [
        "items",
        "discounts",
        "subtotal_after_discounts",
        "rounding",
        "amount_paid_after_rounding",
    ]

    REQUIRED_ITEM_KEYS = ["description", "original_line_amount"]

    REQUIRED_DISCOUNT_KEYS = ["description", "discount_amount"]

    def validate(self, receipt: dict[str, Any]) -> dict[str, Any]:
        self._validate_required(receipt)
        self._validate_present(receipt)

        normalised = self._normalise_amounts(receipt)

        self._validate_item_amounts_are_non_negative(normalised)
        self._validate_discount_amounts_are_non_negative(normalised)
        self._validate_subtotal_final(normalised)
        self._validate_extracted_sum_equals_totals(normalised)

        return normalised

    def _validate_required(self, receipt: dict[str, Any]) -> None:
        if not all(key in receipt for key in self.REQUIRED_FIELDS):
            raise ValueError("Missing required receipt field")

        if not all(
            all(key in item for key in self.REQUIRED_ITEM_KEYS)
            for item in receipt["items"]
        ):
            raise ValueError("Missing required item field")

        if not all(
            all(key in discount for key in self.REQUIRED_DISCOUNT_KEYS)
            for discount in receipt["discounts"]
        ):
            raise ValueError("Missing required discount field")

    def _validate_present(self, receipt: dict[str, Any]) -> None:
        if not all(receipt[key] is not None for key in self.AGGREGATE_AMOUNT_FIELDS):
            raise ValueError("Required monetary field is null")

        if not all(
            all(item[key] is not None for key in self.REQUIRED_ITEM_KEYS)
            for item in receipt["items"]
        ):
            raise ValueError("Required item field is null")

        if not all(
            all(discount[key] is not None for key in self.REQUIRED_DISCOUNT_KEYS)
            for discount in receipt["discounts"]
        ):
            raise ValueError("Required discount field is null")

    def _normalise_amounts(self, receipt: dict[str, Any]) -> dict[str, Any]:
        r = deepcopy(receipt)

        r["items"] = [
            {
                **item,
                "original_line_amount": Decimal(str(item["original_line_amount"])),
            }
            for item in r["items"]
        ]

        r["discounts"] = [
            {
                **discount,
                "discount_amount": Decimal(str(discount["discount_amount"])),
            }
            for discount in r["discounts"]
        ]

        r["subtotal_after_discounts"] = Decimal(str(r["subtotal_after_discounts"]))

        r["rounding"] = Decimal(str(r["rounding"]))

        r["amount_paid_after_rounding"] = Decimal(str(r["amount_paid_after_rounding"]))

        return r

    def _validate_item_amounts_are_non_negative(self, receipt: dict[str, Any]) -> None:
        if not all(item["original_line_amount"] >= 0 for item in receipt["items"]):
            raise ValueError("Item amounts cannot be negative")

    def _validate_discount_amounts_are_non_negative(
        self, receipt: dict[str, Any]
    ) -> None:
        if not all(
            discount["discount_amount"] >= 0 for discount in receipt["discounts"]
        ):
            raise ValueError("Discount amounts cannot be negative")

    def _validate_subtotal_final(self, receipt: dict[str, Any]) -> None:
        if receipt["subtotal_after_discounts"] < 0:
            raise ValueError("Subtotal cannot be negative")

        if receipt["amount_paid_after_rounding"] < 0:
            raise ValueError("Final amount paid cannot be negative")

    def _validate_extracted_sum_equals_totals(self, receipt: dict[str, Any]) -> None:
        item_total = sum(item["original_line_amount"] for item in receipt["items"])

        discount_total = sum(
            discount["discount_amount"] for discount in receipt["discounts"]
        )

        expected_subtotal = item_total - discount_total

        if expected_subtotal != receipt["subtotal_after_discounts"]:
            raise ValueError(
                "Extracted items and discounts do not reconcile to subtotal"
            )

        expected_final = receipt["subtotal_after_discounts"] + receipt["rounding"]

        if expected_final != receipt["amount_paid_after_rounding"]:
            raise ValueError("Subtotal and rounding do not reconcile to final amount")
