# Portable Air-Gapped Hybrid OCR (`airgapped-ocr`)

A **100% offline, self-contained, zero-install OCR and document conversion suite** for **Windows x64 (CPU-only, no administrator rights required)**.

Converts scanned images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`) and multi-page `.pdf` documents into:
- **HTML (`.html`)**: Standalone styled HTML5 documents preserving native `rowspan` and `colspan` table cells (with headerless numeric table support).
- **CSV (`.csv`)**: UTF-8 with BOM (`utf-8-sig`) spreadsheets with vertically/horizontally merged cells **filled down** across spanned rows so every row is a complete record.
- **Markdown (`.md`)**: Full document structure (headings, paragraphs, lists) and GitHub-Flavored Markdown (GFM) pipe tables with filled-down merged cells.

---

## Hybrid 3-Stage Architecture

1. **IBM Docling Offline Pipeline** (`DocLayNet` + `TableFormer ACCURATE` + `RapidOCR` ONNX):
   - Handles general printed documents, multi-column layouts, borderless invoice tables, and automatic light-gray watermark suppression (e.g., diagonal `"SAMPLE"` stamps).
2. **OpenCV Morphological Grid Detector + Cell Refiner**:
   - Detects and perspective-rectifies dense bordered engineering/measurement tables, reconstructs nested `rowspan`/`colspan` hierarchies, reads handwritten numbers, and ignores crossed-out/scribbled-over values in favor of corrected entries.
3. **Local Vision-Language Model (`Qwen3.5-4B` via portable `llama-server.exe`)**:
   - Runs 100% offline on CPU (SSE4.2 / AVX2 / AVX-512 multi-CPU dispatch).
   - Automatically engages whenever `page_needs_vlm_refinement` detects handwriting, partial vertical-divider tables without horizontal borders, pipe/CJK OCR garble, or low printed-OCR confidence—or on demand via `--mode vlm`.

---

## Repository Structure

```text
airgapped-ocr/
├── airgapped_ocr_sandbox.wsb       # Windows Sandbox config (Networking=Disable) for offline testing
├── scripts/
│   └── build_portable.ps1          # One-step staging builder to populate runtime/ and models/
└── portable_ocr/                   # Self-contained folder to copy onto USB / air-gapped PC
    ├── run_ocr.bat                 # Zero-install portable Windows launcher (relative paths only)
    ├── requirements.txt            # Pinned Python dependencies (CPU PyTorch, Docling, RapidOCR)
    ├── README.md                   # Offline usage manual bundled inside portable_ocr
    ├── input/                      # Default drop folder for .jpg / .png / .pdf files
    ├── output/                     # Default destination for .md, .html, and .csv outputs
    ├── runtime/                    # [Built via build_portable.ps1] Relocatable Python 3.11 + VC++ DLLs + llama.cpp
    ├── models/                     # [Built via build_portable.ps1] Pre-fetched Docling + Qwen3.5-4B GGUF weights
    └── src/
        ├── config.py               # Offline env enforcement & MSVC/OpenMP DLL preloader
        ├── pdf_loader.py           # Unicode-safe image loader & PyMuPDF 250 DPI PDF renderer
        ├── docling_pipeline.py     # Offline Docling layout, watermark suppression & table extractor
        ├── grid_vlm_refiner.py     # OpenCV grid rectifier + Local VLM handwriting/table refiner
        ├── exporters.py            # HTML (rowspan/colspan), CSV (UTF-8 BOM fill-down), GFM Markdown
        └── cli.py                  # CLI & batch processor
```

---

## Building the Offline Bundle (Online Staging PC)

Because GitHub excludes multi-gigabyte runtimes and model weights (`portable_ocr/runtime/` and `portable_ocr/models/`), run the staging script once on an internet-connected Windows x64 machine:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_portable.ps1
```

Then copy the resulting `portable_ocr\` directory onto a USB drive and move it to any path on your air-gapped Windows x64 computer—**no installation, no Visual C++ Redistributable installer, and no admin rights needed**.

---

## Usage on the Air-Gapped Computer

### 1. Drop Folder or Drag-and-Drop
Place `.jpg`, `.png`, or `.pdf` files in `portable_ocr\input\` and double-click `portable_ocr\run_ocr.bat` (or drag and drop files onto `run_ocr.bat`). Outputs appear in `portable_ocr\output\`.

### 2. Command Line
```bat
REM Process all files in portable_ocr\input\ (default auto hybrid mode):
portable_ocr\run_ocr.bat

REM Process specific files or folders:
portable_ocr\run_ocr.bat "scan1.jpg" "report.pdf" --output-dir ".\output"

REM Force full-page Vision-Language Model OCR for purely handwritten notes/tables:
portable_ocr\run_ocr.bat "handwritten_card.jpg" --mode vlm

REM Use semicolon delimiter for European/Turkish Excel locales:
portable_ocr\run_ocr.bat "table.jpg" --csv-delimiter ";"
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
| `--mode` | `auto` / `hybrid` (Docling + Grid + auto VLM fallback), `vlm` (force VLM), `docling-only`, or `fast` (no VLM) | `auto` |

---

## Testing Offline in Windows Sandbox

Before transferring to the physical air-gapped machine, double-click `airgapped_ocr_sandbox.wsb` on Windows 10/11 Pro or Enterprise to launch an isolated VM with `<Networking>Disable</Networking>` and zero pre-installed runtimes.
