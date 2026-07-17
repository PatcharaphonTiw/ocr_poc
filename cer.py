#!/usr/bin/env python3
"""
Batch CER (Character Error Rate) evaluation for the chandra-ocr-2 vLLM pipeline.

Reads a ground-truth corpus described by an `index.json` file (see
`extracted_<category>/index.json` for the format: each entry has a `src`
relative path to the source PDF and a `title`; the matching ground-truth text
lives in a sibling `<KEY>.txt` file), runs every source PDF through the same
OCR pipeline as `ocr.py`, computes CER against the ground truth using `jiwer`,
and writes one row per document to a CSV.

Example:
    python cer.py --index path/to/extracted_x/index.json \\
        --pdf-root path/to/raw_pdfs_root \\
        --output cer_results.csv

Requires: pip install jiwer requests pypdfium2 chandra-ocr
"""

import argparse
import csv
import json
import os
import re
import sys
import time

from jiwer import cer as jiwer_cer
from dotenv import load_dotenv

load_dotenv() 

from ocr import get_prompt, ocr_image, pdf_to_images

CSV_FIELDS = ["src", "name", "ground_truth", "OCR_parse", "Times parse", "Character Error Rate(CER)", "Error"]

# Matches the "<!-- ===== page N ===== -->" separators ocr_document inserts
# between pages; these are our own artifacts, not OCR content.
_PAGE_MARKER_RE = re.compile(r"<!--\s*=+\s*page\s+\d+\s*=+\s*-->", re.IGNORECASE)
# Image syntax "![alt text describing the figure](file.webp)" — chandra emits
# these as auto-generated image *descriptions*, not transcribed document text,
# so they're dropped entirely (unlike real links, whose visible text is kept).
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")  # [text](url)
_MD_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_MD_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)
_MD_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$", re.MULTILINE)
_MD_EMPHASIS_RE = re.compile(r"(\*\*\*|\*\*|\*|___|__|_|`)")
_WHITESPACE_RE = re.compile(r"\s+")


def strip_markdown(text: str) -> str:
    """Remove common markdown syntax, leaving just the visible text content."""
    text = _PAGE_MARKER_RE.sub(" ", text)
    text = _MD_IMAGE_RE.sub(" ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_TABLE_SEP_RE.sub(" ", text)
    text = _MD_HEADER_RE.sub("", text)
    text = _MD_BLOCKQUOTE_RE.sub("", text)
    text = _MD_LIST_RE.sub("", text)
    text = _MD_EMPHASIS_RE.sub("", text)
    text = text.replace("|", " ")
    return text


def normalize_text(text: str) -> str:
    """Strip markdown formatting and collapse whitespace, for CER comparison only."""
    if not text:
        return ""
    text = strip_markdown(text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def ocr_document(
    pdf_path: str,
    api_base: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    api_key: str,
    timeout: int,
    dpi: int,
) -> str:
    """OCR every page of `pdf_path` and return the concatenated cleaned text."""
    pages_text = []
    for idx, img in pdf_to_images(pdf_path, dpi=dpi):
        raw = ocr_image(
            img,
            api_base=api_base,
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            api_key=api_key,
            timeout=timeout,
        )
        try:
            from chandra.output import parse_markdown

            cleaned = parse_markdown(raw)
        except Exception:
            cleaned = raw
        pages_text.append(f"\n\n<!-- ===== page {idx + 1} ===== -->\n\n{cleaned}")
    return "".join(pages_text)


def load_index(index_path: str) -> dict:
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ground_truth(gt_dir: str, key: str):
    path = os.path.join(gt_dir, f"{key}.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_processed_srcs(output_path: str) -> set:
    """Return the set of 'src' values already written to an existing output CSV."""
    processed = set()
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        with open(output_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("src"):
                    processed.add(row["src"])
    return processed


def write_error_row(writer, f, src_rel: str, name: str, gt_text, message: str) -> None:
    """Write a CSV row for a document that failed before/during OCR, then flush.

    OCR_parse/Times parse/CER are left blank; `ground_truth` is filled in only
    if it was already available at the point of failure.
    """
    writer.writerow(
        {
            "src": src_rel,
            "name": name,
            "ground_truth": gt_text or "",
            "OCR_parse": "",
            "Times parse": "",
            "Character Error Rate(CER)": "",
            "Error": message,
        }
    )
    f.flush()


def compute_cer(reference: str, hypothesis: str):
    """Return CER (float) computed on normalized text, or None if not computable."""
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    if not ref_norm:
        return None
    try:
        return jiwer_cer(ref_norm, hyp_norm)
    except Exception as e:
        print(f"[warn] CER computation failed: {e}", file=sys.stderr)
        return None


def parse_args():
    ap = argparse.ArgumentParser(
        description="Compute CER of OCR output vs ground truth over an index.json corpus."
    )
    ap.add_argument("--index", required=True, help="Path to the ground-truth index.json")
    ap.add_argument(
        "--gt-dir",
        default=None,
        help="Directory with ground-truth <KEY>.txt files (default: directory of --index)",
    )
    ap.add_argument(
        "--pdf-root",
        required=True,
        help="Root directory used to resolve each index.json entry's 'src' relative path",
    )
    ap.add_argument("-o", "--output", default="cer_results.csv", help="Output CSV path")
    ap.add_argument(
        "--api-base",
        default=os.getenv("ROUTE_PREFIX", "http://localhost:80000/v1"),
        help="vLLM OpenAI-compatible base URL",
    )
    ap.add_argument("--model", default="chandra-ocr-2", help="served-model-name given to vllm serve")
    ap.add_argument(
        "--prompt-type",
        default="ocr_layout",
        choices=["ocr_layout", "ocr"],
        help="chandra prompt type",
    )
    ap.add_argument("--dpi", type=int, default=200, help="Render DPI for PDF pages")
    ap.add_argument("--max-tokens", type=int, default=12384, help="Max output tokens per page")
    ap.add_argument("--api-key", default=os.getenv("API_KEY", "dummy"), help="API key (vLLM ignores by default)")
    ap.add_argument("--timeout", type=int, default=300, help="Per-page request timeout (seconds)")
    ap.add_argument("--keys", default=None, help="Comma-separated list of index keys to process")
    ap.add_argument("--category", default=None, help="Only process entries with this category")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N matching entries")
    ap.add_argument(
        "--start",
        type=int,
        default=None,
        help=(
            "1-based inclusive start position within the filtered entry list "
            "(applied after --keys/--category, before --limit)"
        ),
    )
    ap.add_argument(
        "--end",
        type=int,
        default=None,
        help=(
            "1-based inclusive end position within the filtered entry list "
            "(applied after --keys/--category, before --limit)"
        ),
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing --output CSV and start fresh (overwrite)",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    gt_dir = args.gt_dir or os.path.dirname(os.path.abspath(args.index))
    index = load_index(args.index)

    keys = list(index.keys())
    if args.keys:
        wanted = {k.strip() for k in args.keys.split(",") if k.strip()}
        keys = [k for k in keys if k in wanted]
    if args.category:
        keys = [k for k in keys if index[k].get("category") == args.category]

    if args.start is not None or args.end is not None:
        total = len(keys)
        start = args.start if args.start is not None else 1
        end = args.end if args.end is not None else total
        if start < 1:
            print(f"[error] --start must be >= 1 (got {start})", file=sys.stderr)
            sys.exit(1)
        if end < start:
            print(f"[error] --end ({end}) must be >= --start ({start})", file=sys.stderr)
            sys.exit(1)
        keys = keys[start - 1 : end]
        print(
            f"[info] Selected {len(keys)}/{total} entries (positions {start}-{end})",
            file=sys.stderr,
        )

    if args.limit:
        keys = keys[: args.limit]

    output_exists = os.path.exists(args.output) and os.path.getsize(args.output) > 0
    processed_srcs = set() if args.no_resume else load_processed_srcs(args.output)
    file_mode = "w" if (args.no_resume or not output_exists) else "a"
    write_header = file_mode == "w"

    prompt = get_prompt(args.prompt_type)

    with open(args.output, file_mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for key in keys:
            entry = index[key]
            src_rel = entry["src"]

            if src_rel in processed_srcs:
                print(f"[skip] {key} already in {args.output}", file=sys.stderr)
                continue

            name = entry.get("title", key)
            gt_text = load_ground_truth(gt_dir, key)
            if gt_text is None:
                message = f"no ground-truth file found in {gt_dir}"
                print(f"[warn] {key}: {message}, skipping", file=sys.stderr)
                write_error_row(writer, f, src_rel, name, None, message)
                continue

            pdf_path = os.path.join(args.pdf_root, src_rel)
            if not os.path.exists(pdf_path):
                message = f"PDF not found at {pdf_path}"
                print(f"[warn] {key}: {message}, skipping", file=sys.stderr)
                write_error_row(writer, f, src_rel, name, gt_text, message)
                continue

            print(f"[info] Processing {key} ({name}) ...", file=sys.stderr)
            start = time.perf_counter()
            try:
                ocr_text = ocr_document(
                    pdf_path,
                    api_base=args.api_base,
                    model_name=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    api_key=args.api_key,
                    timeout=args.timeout,
                    dpi=args.dpi,
                )
            except Exception as e:
                print(f"[error] {key}: OCR failed: {e}", file=sys.stderr)
                write_error_row(writer, f, src_rel, name, gt_text, f"OCR failed: {e}")
                continue
            elapsed = time.perf_counter() - start

            score = compute_cer(gt_text, ocr_text)

            writer.writerow(
                {
                    "src": src_rel,
                    "name": name,
                    "ground_truth": gt_text,
                    "OCR_parse": ocr_text,
                    "Times parse": f"{elapsed:.3f}",
                    "Character Error Rate(CER)": f"{score:.6f}" if score is not None else "",
                    "Error": "",
                }
            )
            f.flush()
            print(
                f"[done] {key}: Character Error Rate(CER)={score if score is not None else 'N/A'} time={elapsed:.2f}s",
                file=sys.stderr,
            )

    print(f"[done] wrote results to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()