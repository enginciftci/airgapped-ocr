# Portable Air-Gapped Hybrid OCR (`portable_ocr`)

A 100% self-contained, zero-install, air-gapped OCR and document conversion suite for **Windows x64 (CPU-only, no administrator rights required)**.

Converts scanned images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`) and `.pdf` documents into:
- **HTML (`.html`)**: Standalone styled HTML5 documents preserving native `rowspan` and `colspan` table cells.
- **CSV (`.csv`)**: UTF-8 with BOM (`utf-8-sig`) spreadsheets with vertically/horizontally merged cells **filled down** into every spanned row so each row is a complete record.
- **Markdown (`.md`)**: Full document text (headings, paragraphs, lists) and GitHub-Flavored Markdown (GFM) pipe tables with filled-down merged cells.

---

## Architecture Inside This Portable Folder

```text
portable_ocr/
├── run_ocr.bat               # Main portable Windows launcher (relative paths only)
├── README.md                 # Offline documentation
├── input/                    # Default drop folder for input .jpg / .png / .pdf files
├── output/                   # Default destination for .md, .html, and .csv outputs
├── runtime/
│   ├── python/               # Relocatable Python 3.11 runtime + offline site-packages
│   └── llama/                # Portable llama.cpp CPU inference server + multi-CPU DLLs
├── models/
│   ├── docling/              # Pre-fetched offline IBM Docling models (DocLayNet, TableFormer, RapidOCR)
│   └── vlm/                  # Local Vision-Language Model (Qwen3.5-4B GGUF + multimodal projector)
└── src/
    ├── config.py             # Offline environment enforcement & relative path resolution
    ├── pdf_loader.py         # Unicode-safe image loader & PyMuPDF 250-300 DPI PDF renderer
    ├── docling_pipeline.py   # Offline Docling layout, text, and borderless table pipeline
    ├── grid_vlm_refiner.py   # OpenCV morphological grid + RapidOCR + Local VLM handwriting/cross-out refiner
    ├── exporters.py          # HTML (native rowspan/colspan), CSV (UTF-8 BOM fill-down), Markdown (GFM)
    └── cli.py                # Command-line interface & batch processor
```

---

## Moving to the Air-Gapped Computer

1. Copy or zip the entire `portable_ocr` folder onto a USB drive or external disk.
2. Paste/extract `portable_ocr` to **any directory or drive** on the target Windows x64 computer (e.g., `D:\portable_ocr` or `C:\Users\...\Desktop\portable_ocr`).
3. **No installation, no registry keys, no internet connection, and no Administrator rights are required.**

---

## Usage

### 1. Drag-and-Drop or Batch Folder Mode
- Place `.jpg`, `.png`, or `.pdf` files inside `portable_ocr\input\` and double-click `run_ocr.bat`, **or** drag and drop files directly onto `run_ocr.bat`.
- Converted `.md`, `.html`, and `.csv` files are saved in `portable_ocr\output\`.

### 2. Command-Line (CLI) Usage

```bat
REM Process specific files:
portable_ocr\run_ocr.bat "42-47.jpg" "74-79.jpg" "Swiss_QR-Bill_example.jpg" --output-dir ".\output"

REM Process an entire directory of scanned PDFs and JPGs:
portable_ocr\run_ocr.bat --input-dir "D:\Scans" --output-dir "D:\Converted"

REM Use semicolon delimiter for Turkish Excel locale:
portable_ocr\run_ocr.bat "42-47.jpg" --csv-delimiter ";"

REM Select specific output formats or pipeline modes (use --mode vlm to force full-page VLM OCR):
portable_ocr\run_ocr.bat "document.pdf" --formats md,html,csv --mode auto
```

### CLI Options
| Flag | Description | Default |
| :--- | :--- | :--- |
| `inputs` | One or more file paths or directories to convert | `portable_ocr\input\` |
| `--input-dir` | Directory containing input `.jpg`/`.png`/`.pdf` files | `None` |
| `--output-dir` | Directory where `.md`, `.html`, `.csv` files are written | `portable_ocr\output\` |
| `--formats` | Comma-separated output formats (`md,html,csv`) | `md,html,csv` |
| `--csv-delimiter` | Delimiter character for CSV export (`,` or `;`) | `,` |
| `--no-fill-down` | Leave spanned rows below the first cell of a merge blank in CSV/MD | `False` (Fill-down enabled) |
| `--mode` | `auto` / `hybrid` (Docling + Grid + auto VLM fallback for handwriting/partial grids), `vlm` (force full-page VLM), `docling-only`, or `fast` (no VLM) | `auto` |
