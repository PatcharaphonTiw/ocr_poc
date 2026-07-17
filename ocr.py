#!/usr/bin/env python3
"""
Test script for a self-hosted chandra-ocr-2 vLLM server.

Reads a PDF, renders each page to an image, and POSTs directly to the
OpenAI-compatible endpoint  /v1/chat/completions  that `vllm serve` exposes.

Server was started with something like:
    vllm serve datalab-to/chandra-ocr-2 --served-model-name chandra --port 8000

Requires only lightweight deps (no torch / no GPU on the client side):
    pip install chandra-ocr requests pypdfium2
"""

import argparse
import base64
import io
import sys
import requests

# --- rendering PDF -> images -------------------------------------------------
# pypdfium2 comes as a dependency of chandra-ocr (base install).
import pypdfium2 as pdfium
from PIL import Image


# --- getting the CORRECT prompt ---------------------------------------------
# chandra was trained with a specific prompt per prompt_type. Reuse the package's
# own prompt so we match the training template exactly. If chandra isn't installed,
# we fall back to a minimal prompt (works, but layout/quality may differ).
def get_prompt(prompt_type: str = "ocr_layout") -> str:
    try:
        # The package exposes the prompt templates it uses internally.
        from chandra.prompts import get_prompt as _cp  # newer versions
        return _cp(prompt_type)
    except Exception:
        pass
    try:
        from chandra.prompts import PROMPT_MAPPING  # dict form
        val = PROMPT_MAPPING[prompt_type]
        return val() if callable(val) else val
    except Exception:
        # Minimal fallback. If you can, install chandra-ocr so the exact
        # trained prompt is used instead of this.
        print(
            "[warn] Could not import chandra prompts; using fallback prompt. "
            "Install `chandra-ocr` for the exact trained template.",
            file=sys.stderr,
        )
        return "Convert this image to markdown, preserving the full layout."


def pdf_to_images(pdf_path: str, dpi: int = 200):
    """Yield (page_index, PIL.Image) for each page of the PDF."""
    pdf = pdfium.PdfDocument(pdf_path)
    scale = dpi / 72.0
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            img = bitmap.to_pil().convert("RGB")
            yield i, img
    finally:
        pdf.close()


def image_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def ocr_image(
    img: Image.Image,
    api_base: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    api_key: str = "dummy",
    timeout: int = 300,
) -> str:
    """Send one image to /v1/chat/completions and return the raw text output."""
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(img)},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser(description="Test chandra-ocr-2 vLLM server on a PDF.")
    ap.add_argument("pdf", help="Path to input PDF file")
    ap.add_argument("-o", "--output", default="output.md", help="Output markdown file")
    ap.add_argument(
        "--api-base",
        default="http://10.10.10.187:4000/v1",
        help="vLLM OpenAI-compatible base URL",
    )
    ap.add_argument(
        "--model", default="chandra-ocr-2", help="served-model-name given to vllm serve"
    )
    ap.add_argument(
        "--prompt-type",
        default="ocr_layout",
        choices=["ocr_layout", "ocr"],
        help="chandra prompt type",
    )
    ap.add_argument("--dpi", type=int, default=200, help="Render DPI for PDF pages")
    ap.add_argument(
        "--max-tokens", type=int, default=12384, help="Max output tokens per page"
    )
    ap.add_argument("--api-key", default="dummy", help="API key (vLLM ignores by default)")
    args = ap.parse_args()

    prompt = get_prompt(args.prompt_type)

    pages_md = []
    for idx, img in pdf_to_images(args.pdf, dpi=args.dpi):
        print(f"[info] OCR page {idx + 1} ...", file=sys.stderr)
        raw = ocr_image(
            img,
            api_base=args.api_base,
            model_name=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            api_key=args.api_key,
        )
        # `raw` is the model's raw output. chandra normally post-processes this
        # (parse_markdown) to strip layout tags into clean markdown. If you have
        # chandra installed you can clean it up:
        try:
            from chandra.output import parse_markdown

            cleaned = parse_markdown(raw)
        except Exception:
            cleaned = raw
        pages_md.append(f"\n\n<!-- ===== page {idx + 1} ===== -->\n\n{cleaned}")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("".join(pages_md))

    print(f"[done] wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()