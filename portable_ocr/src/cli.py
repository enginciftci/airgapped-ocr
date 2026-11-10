"""Command-Line Interface for Portable Air-Gapped Hybrid OCR (Docling + Grid + VLM)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import config
from docling_pipeline import build_docling_converter, convert_with_docling
from exporters import (
    DocBlock,
    DocumentConversionResult,
    StructuredTable,
    export_csv,
    export_html,
    export_markdown,
)
from grid_vlm_refiner import (
    LocalVlmServer,
    convert_page_with_vlm,
    detect_and_rectify_bordered_table,
    page_needs_vlm_refinement,
    refine_bordered_table_with_vlm_and_ocr,
)
from pdf_loader import load_document_pages_bgr


def collect_input_files(inputs: list[str], input_dir: Path | None) -> list[Path]:
    """Gather all supported image and PDF files from CLI args or input directory."""
    files: list[Path] = []
    seen: set[Path] = set()

    def _add_file(p: Path) -> None:
        resolved = p.resolve()
        if (
            resolved.is_file()
            and resolved.suffix.lower() in config.SUPPORTED_EXTENSIONS
            and resolved not in seen
        ):
            seen.add(resolved)
            files.append(resolved)

    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            for child in sorted(p.iterdir()):
                _add_file(child)
        elif p.is_file():
            _add_file(p)

    if input_dir is not None and input_dir.is_dir():
        for child in sorted(input_dir.iterdir()):
            _add_file(child)

    if not files:
        # Default fallback: check portable_ocr/input, then parent workspace directory
        if config.DEFAULT_INPUT_DIR.is_dir():
            for child in sorted(config.DEFAULT_INPUT_DIR.iterdir()):
                _add_file(child)
        if not files:
            parent_dir = config.PORTABLE_ROOT.parent
            for child in sorted(parent_dir.iterdir()):
                _add_file(child)

    return files


def process_document(
    input_path: Path,
    mode: str,
    vlm_server: LocalVlmServer | None,
    docling_holder: dict[str, Any],
) -> DocumentConversionResult:
    """Process a single image or multi-page PDF using the hybrid pipeline."""
    pages = load_document_pages_bgr(input_path, dpi=250)
    page_map = {pno: pbgr for pno, pbgr in pages}

    if mode == "docling-only":
        if "converter" not in docling_holder:
            docling_holder["converter"] = build_docling_converter()
        return convert_with_docling(input_path, converter=docling_holder["converter"])

    if mode == "vlm" and vlm_server is not None and vlm_server.is_available():
        all_blocks: list[DocBlock] = []
        all_tables: list[StructuredTable] = []
        for page_no, page_bgr in pages:
            if len(pages) > 1:
                all_blocks.append(
                    DocBlock(block_type="heading", text=f"Page {page_no}", level=2, page_no=page_no)
                )
            p_blocks, p_tables = convert_page_with_vlm(page_bgr, vlm_server, page_no=page_no)
            all_blocks.extend(p_blocks)
            all_tables.extend(p_tables)
        return DocumentConversionResult(
            source_path=input_path,
            blocks=all_blocks,
            tables=all_tables,
        )

    # Check each page for dense bordered grid tables (e.g. 42-47.jpg, 74-79.jpg)
    grid_pages: dict[int, StructuredTable] = {}
    has_non_grid_page = False

    for page_no, page_bgr in pages:
        is_grid, rect_bgr, x_bounds, y_bounds = detect_and_rectify_bordered_table(
            page_bgr
        )
        if is_grid and rect_bgr is not None:
            tbl = refine_bordered_table_with_vlm_and_ocr(
                page_bgr,
                rect_bgr,
                x_bounds,
                y_bounds,
                vlm_server=vlm_server if mode in {"auto", "hybrid"} else None,
                page_no=page_no,
            )
            grid_pages[page_no] = tbl
        else:
            has_non_grid_page = True

    if not has_non_grid_page and grid_pages:
        blocks: list[DocBlock] = []
        tables: list[StructuredTable] = []
        for pno in sorted(grid_pages.keys()):
            tbl = grid_pages[pno]
            if len(grid_pages) > 1:
                blocks.append(
                    DocBlock(block_type="heading", text=f"Page {pno}", level=2, page_no=pno)
                )
            blocks.append(DocBlock(block_type="table", table=tbl, page_no=pno))
            tables.append(tbl)
        return DocumentConversionResult(
            source_path=input_path,
            blocks=blocks,
            tables=tables,
        )

    # Run Docling for general document / borderless table pages (e.g. Swiss_QR-Bill_example.jpg)
    if "converter" not in docling_holder:
        docling_holder["converter"] = build_docling_converter()
    doc_res = convert_with_docling(input_path, converter=docling_holder["converter"])

    # Evaluate each non-grid page with the OCR quality gate; if handwritten or garbled,
    # automatically refine that page via Local VLM when available
    merged_blocks: list[DocBlock] = []
    merged_tables: list[StructuredTable] = []
    max_page = max(pno for pno, _ in pages)
    any_page_overridden = bool(grid_pages)

    for pno in range(1, max_page + 1):
        if pno in grid_pages:
            tbl = grid_pages[pno]
            if max_page > 1:
                merged_blocks.append(
                    DocBlock(block_type="heading", text=f"Page {pno}", level=2, page_no=pno)
                )
            merged_blocks.append(DocBlock(block_type="table", table=tbl, page_no=pno))
            merged_tables.append(tbl)
            continue

        page_blocks = [b for b in doc_res.blocks if b.page_no == pno]
        page_bgr = page_map.get(pno)

        if (
            mode in {"auto", "hybrid"}
            and vlm_server is not None
            and vlm_server.is_available()
            and page_bgr is not None
            and page_needs_vlm_refinement(page_bgr, page_blocks)
        ):
            vlm_blocks, vlm_tables = convert_page_with_vlm(
                page_bgr, vlm_server, page_no=pno
            )
            if vlm_blocks:
                any_page_overridden = True
                if max_page > 1:
                    merged_blocks.append(
                        DocBlock(block_type="heading", text=f"Page {pno}", level=2, page_no=pno)
                    )
                merged_blocks.extend(vlm_blocks)
                merged_tables.extend(vlm_tables)
                continue

        merged_blocks.extend(page_blocks)
        for b in page_blocks:
            if b.block_type == "table" and b.table is not None:
                merged_tables.append(b.table)

    if not any_page_overridden:
        return doc_res

    return DocumentConversionResult(
        source_path=input_path,
        blocks=merged_blocks,
        tables=merged_tables,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Portable Air-Gapped Hybrid OCR (Docling + OpenCV Grid + Local VLM)"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input image (.jpg, .png) or .pdf files, or directories.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing input files to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.DEFAULT_OUTPUT_DIR,
        help="Directory where .md, .html, and .csv outputs will be saved.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="md,html,csv",
        help="Comma-separated list of output formats: md,html,csv (default: md,html,csv).",
    )
    parser.add_argument(
        "--csv-delimiter",
        type=str,
        default=",",
        help="CSV field delimiter (default: ','). Use ';' for Turkish Excel locale if desired.",
    )
    parser.add_argument(
        "--no-fill-down",
        action="store_true",
        help="Do not repeat vertically/horizontally merged cell values on spanned rows in CSV/MD.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "hybrid", "vlm", "docling-only", "fast"],
        default="auto",
        help="Pipeline mode: 'auto'/'hybrid' (Docling + Grid + VLM), 'vlm' (force VLM), 'docling-only', or 'fast'.",
    )

    args = parser.parse_args(argv)
    files = collect_input_files(args.inputs, args.input_dir)
    if not files:
        print("[!] No input files (.jpg, .png, .pdf) found to process.", file=sys.stderr)
        return 1

    formats = {f.strip().lower() for f in args.formats.split(",") if f.strip()}
    fill_down = not args.no_fill_down
    out_dir: Path = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vlm_server: LocalVlmServer | None = None
    if args.mode in {"auto", "hybrid", "vlm"}:
        vlm_server = LocalVlmServer()

    docling_holder: dict[str, Any] = {}

    print(f"=== Portable Air-Gapped OCR Pipeline ({len(files)} file(s)) ===")
    print(f"Output directory: {out_dir}")

    try:
        for idx, file_path in enumerate(files, start=1):
            t0 = time.time()
            print(f"\n[{idx}/{len(files)}] Processing: {file_path.name} ...")
            result = process_document(
                file_path,
                mode=args.mode,
                vlm_server=vlm_server,
                docling_holder=docling_holder,
            )
            stem = file_path.stem
            written: list[Path] = []

            if "md" in formats or "markdown" in formats:
                md_path = export_markdown(result, out_dir / f"{stem}.md", fill_down=fill_down)
                written.append(md_path)
            if "html" in formats:
                html_path = export_html(result, out_dir / f"{stem}.html")
                written.append(html_path)
            if "csv" in formats:
                csv_paths = export_csv(
                    result,
                    out_dir / f"{stem}.csv",
                    delimiter=args.csv_delimiter,
                    fill_down=fill_down,
                )
                written.extend(csv_paths)

            elapsed = time.time() - t0
            print(f"    Completed in {elapsed:.1f}s -> {', '.join(p.name for p in written)}")
    finally:
        if vlm_server is not None:
            vlm_server.stop()

    print("\nAll files processed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
