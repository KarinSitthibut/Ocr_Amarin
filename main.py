import base64
import json
import re
import time
from pathlib import Path

import fitz  # pymupdf
import io
from PIL import Image, ImageEnhance
from openai import AzureOpenAI

from config import (API_KEY, API_VERSION, ENDPOINT, MODEL_NAME,
                    PRICE_INPUT_PER_1M, PRICE_OUTPUT_PER_1M, USD_TO_THB)
from prompt import SYSTEM_PROMPT, USER_PROMPT

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")

FIELDS = [
    "Vender_Name", "Name", "Address", "Telephone", "Term_of_Payment",
    "VAT_Registration_No", "Bank_Name", "Branch_Name", "Bank_Account",
    "Method_of_Payment", "County", "Email", "Mobile_Phone", "Extension",
    "Search_Name", "Branch_Office", "Branch_Office_Number",
]


def preprocess_image(img_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(img_bytes))
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def split_image(img_bytes: bytes) -> list[bytes]:
    img = Image.open(io.BytesIO(img_bytes))
    w, h = img.size
    halves = [
        img.crop((0, 0, w, h // 2)),
        img.crop((0, h // 2, w, h)),
    ]
    result = []
    for half in halves:
        buf = io.BytesIO()
        half.save(buf, format="PNG")
        result.append(buf.getvalue())
    return result


def pdf_to_images(pdf_path: Path) -> list[tuple[bytes, str]]:
    doc = fitz.open(str(pdf_path))
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img_bytes = preprocess_image(pix.tobytes("png"))
        for half in split_image(img_bytes):
            images.append((half, "image/png"))
    doc.close()
    return images


def jpeg_to_images(jpeg_path: Path) -> list[tuple[bytes, str]]:
    with open(jpeg_path, "rb") as f:
        return [(preprocess_image(f.read()), "image/png")]


def parse_json_response(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw.strip())


def normalize(data: dict) -> dict:
    return {field: data.get(field, "") for field in FIELDS}


def calculate_cost_thb(prompt_tokens: int, completion_tokens: int) -> float:
    cost_usd = (prompt_tokens / 1_000_000 * PRICE_INPUT_PER_1M
                + completion_tokens / 1_000_000 * PRICE_OUTPUT_PER_1M)
    return cost_usd * USD_TO_THB


def extract_data(client: AzureOpenAI, images: list[tuple[bytes, str]]) -> tuple[dict, int, int]:
    content = [{"type": "text", "text": USER_PROMPT}]
    for img_bytes, mime_type in images:
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64}", "detail": "high"},
        })

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )

    raw = response.choices[0].message.content
    usage = response.usage
    return normalize(parse_json_response(raw)), usage.prompt_tokens, usage.completion_tokens


def process_file(client: AzureOpenAI, file_path: Path) -> tuple[int, int]:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        images = pdf_to_images(file_path)
    elif suffix in (".jpg", ".jpeg"):
        images = jpeg_to_images(file_path)
    else:
        return 0, 0

    data, prompt_tokens, completion_tokens = extract_data(client, images)

    output_path = OUTPUT_DIR / (file_path.stem + ".json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return prompt_tokens, completion_tokens


def main():
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    files = (
        list(INPUT_DIR.glob("*.pdf"))
        + list(INPUT_DIR.glob("*.jpg"))
        + list(INPUT_DIR.glob("*.jpeg"))
    )

    if not files:
        print("No files found in input/")
        return

    print(f"Found {len(files)} file(s)\n")

    client = AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        api_version=API_VERSION,
    )

    total_prompt = total_completion = 0
    start_all = time.perf_counter()

    for file_path in files:
        print(f"  {file_path.name} ... ", end="", flush=True)
        start = time.perf_counter()
        try:
            prompt_tokens, completion_tokens = process_file(client, file_path)
            elapsed = time.perf_counter() - start
            cost = calculate_cost_thb(prompt_tokens, completion_tokens)
            total_tokens = prompt_tokens + completion_tokens
            total_prompt += prompt_tokens
            total_completion += completion_tokens
            print(f"done  |  {elapsed:.1f}s  |  tokens: {total_tokens:,}  |  {cost:.4f} THB")
        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"error ({elapsed:.1f}s): {e}")

    total_elapsed = time.perf_counter() - start_all
    total_cost = calculate_cost_thb(total_prompt, total_completion)
    print(f"\nTotal: {total_elapsed:.1f}s  |  {total_prompt + total_completion:,} tokens  |  {total_cost:.4f} THB")
    print("Done")


if __name__ == "__main__":
    main()
