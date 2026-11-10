"""Docling Offline Document & Table Extraction Pipeline."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np

import config  # noqa: F401 - ensures offline env vars are set first
from exporters import DocBlock, DocumentConversionResult, StructuredTable, TableCell
from pdf_loader import load_image_bgr


def build_docling_converter(artifacts_dir: Path | None = None) -> Any:
    """Create an offline Docling DocumentConverter for PDF and Image inputs."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        RapidOcrOptions,
        TableFormerMode,
        TableStructureOptions,
    )
    from docling.document_converter import (
        DocumentConverter,
        ImageFormatOption,
        PdfFormatOption,
    )

    resolved_artifacts = artifacts_dir or config.DOCLING_MODELS_DIR

    ocr_opts = RapidOcrOptions(
        force_full_page_ocr=True,
    )
    table_opts = TableStructureOptions(
        do_cell_matching=True,
        mode=TableFormerMode.ACCURATE,
    )

    pipeline_options = PdfPipelineOptions(
        artifacts_path=str(resolved_artifacts) if resolved_artifacts.exists() else None,
        do_ocr=True,
        do_table_structure=True,
        ocr_options=ocr_opts,
        table_structure_options=table_opts,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        }
    )
    return converter


def _suppress_light_watermark_if_present(img_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Suppress large light-gray watermarks (e.g. 'SAMPLE') on white-background documents."""
    # Check if background is predominantly white (median pixel intensity > 240)
    if float(np.median(img_bgr)) < 235.0:
        return img_bgr, False

    b, g, r = cv2.split(img_bgr)
    max_c = cv2.max(cv2.max(b, g), r)
    min_c = cv2.min(cv2.min(b, g), r)

    # Light-gray watermark pixels: neutral gray (max - min < 20) with brightness in [155, 238]
    wm_mask = (min_c >= 155) & (max_c <= 238) & ((max_c - min_c) < 20)
    wm_ratio = float(np.mean(wm_mask))
    if wm_ratio < 0.015 or wm_ratio > 0.35:
        return img_bgr, False

    # Dilate slightly to remove anti-aliased halo around watermark letters,
    # while protecting dark text pixels (max_c < 135)
    kernel = np.ones((3, 3), dtype=np.uint8)
    wm_dilated = cv2.dilate(wm_mask.astype(np.uint8) * 255, kernel, iterations=1) > 0
    dark_text_mask = max_c < 135
    clean_mask = wm_dilated & (~dark_text_mask)

    cleaned = img_bgr.copy()
    cleaned[clean_mask] = [255, 255, 255]
    return cleaned, True


def docling_table_to_structured(table_item: Any, page_no: int = 1) -> StructuredTable:
    """Convert a Docling TableItem into a StructuredTable preserving spans."""
    data = getattr(table_item, "data", None)
    if data is None:
        return StructuredTable(num_rows=0, num_cols=0, page_no=page_no)

    num_rows = int(getattr(data, "num_rows", 0))
    num_cols = int(getattr(data, "num_cols", 0))
    cells: list[TableCell] = []

    for tc in getattr(data, "table_cells", []):
        r_start = int(getattr(tc, "start_row_offset_idx", 0))
        r_end = int(getattr(tc, "end_row_offset_idx", r_start + 1))
        c_start = int(getattr(tc, "start_col_offset_idx", 0))
        c_end = int(getattr(tc, "end_col_offset_idx", c_start + 1))
        text = str(getattr(tc, "text", "") or "").strip()
        is_header = bool(getattr(tc, "column_header", False)) or (r_start == 0)

        cells.append(
            TableCell(
                row_start=r_start,
                row_end=max(r_start + 1, r_end),
                col_start=c_start,
                col_end=max(c_start + 1, c_end),
                text=text,
                is_header=is_header,
            )
        )

    return StructuredTable(
        num_rows=num_rows,
        num_cols=num_cols,
        cells=cells,
        page_no=page_no,
    )


def _item_sort_key(item: Any, order_idx: int) -> tuple[int, float, float, int]:
    """Compute (page_no, top_down_y_band, left_x, fallback_order) for natural reading order."""
    prov = getattr(item, "prov", None)
    if not prov or len(prov) == 0:
        return (1, 99999.0, 0.0, order_idx)

    p0 = prov[0]
    page_no = int(getattr(p0, "page_no", 1))
    bbox = getattr(p0, "bbox", None)
    if bbox is None:
        return (page_no, 99999.0, 0.0, order_idx)

    # Docling commonly uses BOTTOMLEFT origin where larger t is higher on the page
    origin = str(getattr(bbox, "coord_origin", "BOTTOMLEFT"))
    t_val = float(getattr(bbox, "t", 0.0))
    l_val = float(getattr(bbox, "l", 0.0))

    top_down_y = -t_val if "BOTTOMLEFT" in origin else t_val
    # Quantize Y slightly (8 pt bands) so side-by-side columns read left-to-right
    y_band = round(top_down_y / 8.0) * 8.0
    return (page_no, y_band, l_val, order_idx)


def convert_with_docling(
    input_path: Path,
    converter: Any | None = None,
) -> DocumentConversionResult:
    """Run Docling conversion on an image or PDF and extract blocks and tables."""
    from docling_core.types.doc import (
        ListItem,
        SectionHeaderItem,
        TableItem,
        TextItem,
        TitleItem,
    )

    if converter is None:
        converter = build_docling_converter()

    tmp_file: Path | None = None
    target_path = input_path

    # If input is an image with a light-gray watermark (e.g., 'SAMPLE'), suppress watermark first
    if input_path.suffix.lower() in config.SUPPORTED_IMAGE_EXTENSIONS:
        try:
            img_bgr = load_image_bgr(input_path)
            cleaned_bgr, was_cleaned = _suppress_light_watermark_if_present(img_bgr)
            if was_cleaned:
                fd, tmp_name = tempfile.mkstemp(suffix=".png", prefix="docling_clean_")
                import os

                os.close(fd)
                tmp_file = Path(tmp_name)
                cv2.imwrite(str(tmp_file), cleaned_bgr)
                target_path = tmp_file
        except Exception:
            target_path = input_path

    try:
        conv_res = converter.convert(str(target_path))
    finally:
        if tmp_file is not None and tmp_file.exists():
            try:
                tmp_file.unlink()
            except Exception:
                pass

    doc = conv_res.document

    # Gather items including any text nested inside PictureItems (traverse_pictures=True)
    raw_items: list[tuple[int, Any, int]] = []
    for idx, (item, level) in enumerate(doc.iterate_items(traverse_pictures=True)):
        raw_items.append((idx, item, level))

    # Sort items in top-to-bottom, left-to-right page reading order
    raw_items.sort(key=lambda entry: _item_sort_key(entry[1], entry[0]))

    blocks: list[DocBlock] = []
    tables: list[StructuredTable] = []

    for _idx, item, level in raw_items:
        page_no = 1
        prov = getattr(item, "prov", None)
        if prov and len(prov) > 0:
            page_no = int(getattr(prov[0], "page_no", 1))

        if isinstance(item, TitleItem):
            text = (item.text or "").strip()
            if text:
                blocks.append(
                    DocBlock(block_type="heading", text=text, level=1, page_no=page_no)
                )
        elif isinstance(item, SectionHeaderItem):
            text = (item.text or "").strip()
            if text:
                blocks.append(
                    DocBlock(
                        block_type="heading",
                        text=text,
                        level=min(6, max(2, int(level) + 1)),
                        page_no=page_no,
                    )
                )
        elif isinstance(item, ListItem):
            text = (item.text or "").strip()
            if text:
                blocks.append(
                    DocBlock(block_type="text", text=f"- {text}", page_no=page_no)
                )
        elif isinstance(item, TextItem):
            text = (item.text or "").strip()
            if text:
                blocks.append(DocBlock(block_type="text", text=text, page_no=page_no))
        elif isinstance(item, TableItem):
            st = docling_table_to_structured(item, page_no=page_no)
            if st.num_rows > 0 and st.num_cols > 0:
                tables.append(st)
                blocks.append(DocBlock(block_type="table", table=st, page_no=page_no))

    return DocumentConversionResult(
        source_path=input_path,
        blocks=blocks,
        tables=tables,
        docling_markdown=doc.export_to_markdown(),
        docling_html=doc.export_to_html(),
    )
