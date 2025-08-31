#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDF (scanned) ➜ Per-page PNG images ➜ Gemini Flash OCR (Hindi ➜ Markdown)
➜ CSV (page, markdown) + Compiled book_converted.md

Features:
- High-quality page rendering via PyMuPDF (300 DPI default; configurable)
- Resumable (skips already-processed pages in CSV)
- Robust retries with exponential backoff
- Clean Markdown output (page breaks + optional TOC)
- Strict "Markdown only" instruction to Gemini for minimal noise
"""

import os
import io
import sys
import csv
import json
import time
import hashlib
import argparse
from pathlib import Path
from typing import Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image
import pandas as pd
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt

# Google Gemini SDK
import google.generativeai as genai


# -----------------------------
# Utility / IO
# -----------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def natural_sort_key(p: Path):
    # sort "page_2" before "page_10"
    return [int(t) if t.isdigit() else t for t in ''.join(
        c if c.isdigit() else f' {c} ' for c in p.stem).split()]


def page_image_name(page_index: int) -> str:
    return f"page_{page_index+1:03d}.png"


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


# -----------------------------
# PDF ➜ Images
# -----------------------------

def render_pdf_to_images(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 300,
    overwrite: bool = False
) -> int:
    """
    Render each page of the PDF to a PNG at the requested DPI.
    For scanned books this preserves readability (and usually exceeds original).
    """
    ensure_dir(out_dir)
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    total = len(doc)

    for i in tqdm(range(total), desc="Rendering pages", unit="page"):
        img_path = out_dir / page_image_name(i)
        if img_path.exists() and not overwrite:
            continue
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img.save(img_path, format="PNG", optimize=True)
    doc.close()
    return total


# -----------------------------
# Gemini setup
# -----------------------------

def init_gemini(model_name: str, api_key: Optional[str] = None):
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    genai.configure(api_key=api_key)
    # System instruction: force Markdown only, preserve Hindi,
    # remove boilerplate, keep structure, keep headings, lists, quotes.
    system_instruction = (
        "You are an expert OCR agent for scanned Hindi books. "
        "Extract ONLY the text content in **Markdown** (no preamble, no code fences). "
        "Preserve the original language (Hindi/Devanagari). "
        "Preserve headings, lists, quotes, verses, emphasis, and paragraph breaks. "
        "Remove page headers/footers, running titles, and visible page numbers. "
        "Do not translate. Do not summarize. Do not add commentary. "
        "Output must be valid Markdown only."
    )
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction
    )
    return model


@retry(wait=wait_exponential(multiplier=1, min=2, max=30),
       stop=stop_after_attempt(6))
def gemini_markdown_from_image(model, image_path: Path, max_output_tokens: int = 8192) -> str:
    """
    Sends a single image to Gemini and returns Markdown text.
    Retries on transient errors / rate limits.
    """
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # Provide both the image and a short user instruction
    # (system instruction already added when creating model)
    # The SDK accepts "inline data" dict objects for images
    image_part = {
        "mime_type": "image/png",
        "data": img_bytes
    }

    # Gentle nudges for cleanup and OCR focus:
    user_instruction = (
        "Extract the text in Markdown. Keep Hindi text. "
        "Remove artifacts (scanning noise, watermarks) and running headers/footers."
    )

    resp = model.generate_content(
        [image_part, {"text": user_instruction}],
        generation_config={
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": max_output_tokens,
        }
    )
    # Newer SDKs: use .text; older might use .candidates[0].content.parts
    md = (getattr(resp, "text", None) or "").strip()
    if not md:
        # Fall back if structure differs
        try:
            parts = resp.candidates[0].content.parts
            md = "".join(getattr(p, "text", "") for p in parts).strip()
        except Exception:
            pass
    if not md:
        raise RuntimeError("Empty OCR result from Gemini for: " + image_path.name)
    return md


# -----------------------------
# Orchestration
# -----------------------------

def process_book(
    pdf: Path,
    work_dir: Path,
    model_name: str = "gemini-2.0-flash",
    dpi: int = 300,
    overwrite_images: bool = False,
    resume: bool = True,
    csv_name: str = "pages_markdown.csv",
    md_name: str = "book_converted.md",
    add_toc: bool = True
):
    images_dir = work_dir / "images"
    ensure_dir(work_dir)
    ensure_dir(images_dir)

    # 1) Render images
    total_pages = render_pdf_to_images(pdf, images_dir, dpi=dpi, overwrite=overwrite_images)

    # 2) Init Gemini
    model = init_gemini(model_name)

    # 3) CSV (resumable)
    csv_path = work_dir / csv_name
    existing = {}
    if csv_path.exists() and resume:
        df_existing = pd.read_csv(csv_path)
        for _, row in df_existing.iterrows():
            existing[int(row["page"])] = row["markdown"]

    rows = []
    if existing:
        # preserve existing rows in-memory
        for p, md in existing.items():
            rows.append({"page": p, "image_path": str(images_dir / page_image_name(p-1)), "markdown": md})

    # 4) OCR each page
    for page_idx in tqdm(range(total_pages), desc="OCR via Gemini", unit="page"):
        page_no = page_idx + 1
        img_path = images_dir / page_image_name(page_idx)

        if resume and (page_no in existing):
            continue

        try:
            md = gemini_markdown_from_image(model, img_path)
        except Exception as e:
            print(f"[WARN] Gemini failed on page {page_no}: {e}", file=sys.stderr)
            md = ""  # leave blank but keep row so we can retry later if desired

        rows.append({"page": page_no, "image_path": str(img_path), "markdown": md})

        # Flush to CSV progressively (safe against crashes)
        df = pd.DataFrame(rows).sort_values("page")
        df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

    # 5) Compile Markdown
    compile_markdown(work_dir / md_name, rows, add_toc=add_toc)
    print(f"\n✅ Done!\n- CSV: {csv_path}\n- Markdown book: {work_dir / md_name}\n- Images: {images_dir}")


def compile_markdown(out_md: Path, rows, add_toc: bool = True):
    rows_sorted = sorted(rows, key=lambda r: r["page"])
    buf = []

    # Optional: Table of contents (just page anchors; you can remove if not useful)
    if add_toc:
        buf.append("# विषयसूची (Table of Contents)")
        for r in rows_sorted:
            buf.append(f"- [पृष्ठ {r['page']}](#page-{r['page']})")
        buf.append("\n---\n")

    # Pages
    for r in rows_sorted:
        page = r["page"]
        md = (r["markdown"] or "").strip()
        buf.append(f"\n\n---\n\n<a id='page-{page}'></a>\n**पृष्ठ {page}**\n\n")
        # Keep as-is; many Hindi books have headings etc.
        buf.append(md if md else "_(No text extracted)_")

    out_md.write_text("\n".join(buf), encoding="utf-8")


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert scanned Hindi PDF to Markdown using Gemini Flash (2.x)"
    )
    # PDF path is now hardcoded as requested
    parser.add_argument("--outdir", type=str, default="book_out", help="Working/output directory")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI for page images (300 recommended)")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash",
                        help="Gemini model name (e.g., gemini-2.0-flash or gemini-2.5-flash)")
    parser.add_argument("--overwrite-images", action="store_true", help="Re-render images even if they exist")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from existing CSV")
    parser.add_argument("--no-toc", action="store_true", help="Do not include a table of contents")
    parser.add_argument("--csv", type=str, default="pages_markdown.csv", help="CSV filename")
    parser.add_argument("--md", type=str, default="book_converted.md", help="Output Markdown filename")
    args = parser.parse_args()

    # Hardcoded PDF path as requested
    pdf_path = Path("/Users/apple/Documents/COGNIO/IDEATE/book-maker/book-maker/baba_book.pdf")
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    work_dir = Path(args.outdir).expanduser().resolve()
    process_book(
        pdf=pdf_path,
        work_dir=work_dir,
        model_name=args.model,
        dpi=args.dpi,
        overwrite_images=args.overwrite_images,
        resume=not args.no_resume,
        csv_name=args.csv,
        md_name=args.md,
        add_toc=not args.no_toc
    )


if __name__ == "__main__":
    main()
