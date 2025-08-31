#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to process first 5 pages of PDF
Extracts text from each page and appends into a single markdown file
"""

import os
import sys
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm
import google.generativeai as genai
from tenacity import retry, wait_exponential, stop_after_attempt

# Configuration
PDF_PATH = Path("/Users/apple/Documents/COGNIO/IDEATE/book-maker/book-maker/baba_book.pdf")
OUTPUT_DIR = Path("test_output")
IMAGES_DIR = OUTPUT_DIR / "images"
FINAL_MD = OUTPUT_DIR / "test_5_pages.md"
DPI = 300
MAX_PAGES = 5

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def render_page_to_image(doc, page_idx: int, out_path: Path, dpi: int = 300):
    """Render a single PDF page to PNG"""
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    page = doc.load_page(page_idx)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img.save(out_path, format="PNG", optimize=True)
    print(f"  Rendered page {page_idx + 1} to {out_path.name}")

def init_gemini():
    """Initialize Gemini model"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY is not set in environment.", file=sys.stderr)
        sys.exit(1)
    
    genai.configure(api_key=api_key)
    
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
        model_name="gemini-2.0-flash",
        system_instruction=system_instruction
    )
    return model

@retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(6))
def extract_text_from_image(model, image_path: Path) -> str:
    """Extract text from image using Gemini"""
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    
    image_part = {
        "mime_type": "image/png",
        "data": img_bytes
    }
    
    user_instruction = (
        "Extract the text in Markdown. Keep Hindi text. "
        "Remove artifacts (scanning noise, watermarks) and running headers/footers."
    )
    
    resp = model.generate_content(
        [image_part, {"text": user_instruction}],
        generation_config={
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 8192,
        }
    )
    
    md = (getattr(resp, "text", None) or "").strip()
    if not md:
        try:
            parts = resp.candidates[0].content.parts
            md = "".join(getattr(p, "text", "") for p in parts).strip()
        except Exception:
            pass
    
    if not md:
        raise RuntimeError(f"Empty OCR result from Gemini for: {image_path.name}")
    
    return md

def main():
    print(f"Testing PDF to Markdown conversion for first {MAX_PAGES} pages")
    print(f"PDF: {PDF_PATH}")
    print(f"Output: {FINAL_MD}\n")
    
    # Check PDF exists
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found: {PDF_PATH}", file=sys.stderr)
        sys.exit(1)
    
    # Create directories
    ensure_dir(OUTPUT_DIR)
    ensure_dir(IMAGES_DIR)
    
    # Initialize Gemini
    print("Initializing Gemini model...")
    model = init_gemini()
    
    # Open PDF
    print(f"\nOpening PDF...")
    doc = fitz.open(PDF_PATH)
    total_pages = min(len(doc), MAX_PAGES)
    print(f"Processing {total_pages} pages (PDF has {len(doc)} total pages)\n")
    
    # Initialize markdown content
    markdown_content = []
    markdown_content.append(f"# Test Extraction - First {MAX_PAGES} Pages\n")
    markdown_content.append("This is a test extraction of the first 5 pages to verify the process.\n")
    
    # Process each page
    for page_idx in range(total_pages):
        page_num = page_idx + 1
        print(f"\n--- Processing Page {page_num} ---")
        
        # Step 1: Render page to image
        img_path = IMAGES_DIR / f"page_{page_num:03d}.png"
        render_page_to_image(doc, page_idx, img_path, dpi=DPI)
        
        # Step 2: Extract text using Gemini
        print(f"  Extracting text with Gemini...")
        try:
            extracted_text = extract_text_from_image(model, img_path)
            print(f"  Successfully extracted {len(extracted_text)} characters")
        except Exception as e:
            print(f"  ERROR extracting text: {e}")
            extracted_text = "_(Failed to extract text from this page)_"
        
        # Step 3: Append to markdown
        markdown_content.append(f"\n---\n")
        markdown_content.append(f"\n## Page {page_num}\n")
        markdown_content.append(extracted_text)
        
        # Save progress after each page
        with open(FINAL_MD, "w", encoding="utf-8") as f:
            f.write("\n".join(markdown_content))
        print(f"  Appended to {FINAL_MD}")
    
    doc.close()
    
    # Final summary
    print(f"\n{'='*50}")
    print(f"✅ Test completed successfully!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🖼️  Images saved to: {IMAGES_DIR}")
    print(f"📄 Final markdown: {FINAL_MD}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()