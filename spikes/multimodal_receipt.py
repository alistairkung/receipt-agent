from pathlib import Path

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
import os

from lib.receipt_calculator import ReceiptCalculator
from lib.receipt_extractor import build_receipt_extraction_chain
from lib.receipt_validator import ReceiptValidator
from receipt_agent.pipeline import image_data_url


def extract_receipt():
    load_dotenv()

    api_key = os.environ["DEEPSEEK_API_KEY"]

    llm = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        api_key=api_key,
        max_tokens=6000,
        timeout=30,
        max_retries=2,
        extra_body={"thinking": {"type": "disabled"}},
    )

    image_paths = sorted(Path("public_test").glob("*.jpg"))

    validated_receipts = []

    chain = build_receipt_extraction_chain(llm)
    validator = ReceiptValidator()
    calculator = ReceiptCalculator()

    for image_path in image_paths:
        print(f"\nProcessing {image_path.name}")

        image_url = image_data_url(image_path)
        receipt = chain.invoke({"image_url": image_url})

        try:
            validated = validator.validate(receipt)
            validated_receipts.append(validated)
            print("✓ valid")

        except ValueError as e:
            print(f"✗ validation failed: {e}")
            print(receipt)

    print(f"\nValidated {len(validated_receipts)} " f"of {len(image_paths)} receipts")

    answers = calculator.calculate(validated_receipts)

    print(answers)


if __name__ == "__main__":
    extract_receipt()
