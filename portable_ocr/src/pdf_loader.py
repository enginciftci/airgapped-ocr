"""Image and PDF page loader supporting Unicode/Turkish paths on Windows."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def load_image_bgr(path: Path) -> np.ndarray:
    """Load an image file into a BGR numpy array (safe for non-ASCII Windows paths)."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image file: {path}")
    return img


def load_document_pages_bgr(path: Path, dpi: int = 250) -> list[tuple[int, np.ndarray]]:
    """Return list of (1-based page_no, image_bgr) for an image or multi-page PDF."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pymupdf as fitz

        pages: list[tuple[int, np.ndarray]] = []
        doc = fitz.open(str(path))
        try:
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            for idx, page in enumerate(doc):
                pix = page.get_pixmap(matrix=mat, alpha=False)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                pages.append((idx + 1, bgr))
        finally:
            doc.close()
        return pages

    return [(1, load_image_bgr(path))]
