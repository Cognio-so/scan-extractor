# PDF to Markdown Converter (Hindi Book OCR)

Convert scanned Hindi book PDFs to Markdown using Google Gemini Flash AI model.

## Prerequisites

### 1. Python Environment
- Python 3.8 or higher
- pip package manager

### 2. Google Gemini API Key
Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Installation

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install PyMuPDF pillow pandas tqdm tenacity google-generativeai
```

### Step 2: Set API Key
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

For permanent setup (macOS/Linux):
```bash
echo 'export GOOGLE_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

## Usage

### Basic Run
```bash
python pdf_to_markdown_gemini.py
```

The script will process `baba_book.pdf` (hardcoded path) and:
1. Create `book_out/` directory with:
   - `images/` - PNG files for each page (300 DPI)
   - `pages_markdown.csv` - Page-by-page extracted text
   - `book_converted.md` - Complete book in Markdown

### Command Line Options

```bash
# Specify output directory
python pdf_to_markdown_gemini.py --outdir my_output

# Use higher DPI for better quality (default: 300)
python pdf_to_markdown_gemini.py --dpi 600

# Use Gemini 2.5 Flash model
python pdf_to_markdown_gemini.py --model gemini-2.5-flash

# Re-render images even if they exist
python pdf_to_markdown_gemini.py --overwrite-images

# Start fresh (ignore existing CSV progress)
python pdf_to_markdown_gemini.py --no-resume

# Skip table of contents in output
python pdf_to_markdown_gemini.py --no-toc

# Custom output filenames
python pdf_to_markdown_gemini.py --csv my_pages.csv --md my_book.md
```

## Output Files

### 1. `book_out/images/`
- `page_001.png`, `page_002.png`, ... `page_093.png`
- High-resolution PNG images of each PDF page

### 2. `book_out/pages_markdown.csv`
```csv
page,image_path,markdown
1,book_out/images/page_001.png,"# Chapter 1 ..."
2,book_out/images/page_002.png,"Content of page 2..."
```

### 3. `book_out/book_converted.md`
Complete book with:
- Table of contents (5?7/8B@)
- Page markers (*C7M  1, *C7M  2, etc.)
- All extracted Hindi text in Markdown format

## Features

- **Resumable**: If interrupted, re-run to continue from last processed page
- **High Quality**: 300 DPI rendering preserves text clarity
- **Hindi Support**: Preserves Devanagari script without translation
- **Clean Output**: Removes headers, footers, page numbers
- **Markdown Formatting**: Preserves headings, lists, emphasis
- **Error Recovery**: Automatic retries with exponential backoff

## Troubleshooting

### API Key Not Working
```bash
# Verify key is set
echo $GOOGLE_API_KEY

# Test with simple API call
python -c "import google.generativeai as genai; genai.configure(api_key='$GOOGLE_API_KEY'); print('API key valid')"
```

### Rate Limiting
The script automatically retries with exponential backoff. For large books, consider:
- Using `--no-resume` flag cautiously
- Processing in batches if needed

### Memory Issues
For very large PDFs:
```bash
# Process with lower DPI
python pdf_to_markdown_gemini.py --dpi 200
```

### Missing Dependencies
```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt
```

## Processing Time Estimates
- 93-page book: ~15-30 minutes (depends on API response time)
- Each page: ~10-20 seconds (including API call and retry buffer)

## Notes
- The PDF path is hardcoded to: `/Users/apple/Documents/COGNIO/IDEATE/book-maker/book-maker/baba_book.pdf`
- Gemini Flash models are optimized for speed while maintaining quality
- CSV is updated after each page for crash recovery