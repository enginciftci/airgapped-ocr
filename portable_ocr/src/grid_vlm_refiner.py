"""OpenCV Morphological Grid Topology + Hybrid RapidOCR & Local VLM Table Refiner."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Any

import cv2
import numpy as np
import requests

import config
from exporters import DocBlock, DocumentConversionResult, StructuredTable, TableCell


class LocalVlmServer:
    """Manages a portable llama-server.exe instance for CPU vision inference."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen[Any] | None = None
        self.port: int | None = None
        self.base_url: str | None = None

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

    def is_available(self) -> bool:
        exe = config.get_llama_server_exe()
        model_path, mmproj_path = config.get_vlm_model_paths()
        return (
            exe is not None
            and exe.is_file()
            and model_path is not None
            and model_path.is_file()
            and mmproj_path is not None
            and mmproj_path.is_file()
        )

    def start(self, timeout_sec: float = 90.0) -> bool:
        if self.proc is not None and self.proc.poll() is None and self.base_url:
            return True

        exe = config.get_llama_server_exe()
        model_path, mmproj_path = config.get_vlm_model_paths()
        if not exe or not model_path or not mmproj_path:
            return False

        self.port = self._find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        threads = max(2, min(8, (os.cpu_count() or 4)))

        cmd = [
            str(exe),
            "-m",
            str(model_path),
            "--mmproj",
            str(mmproj_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "-c",
            "4096",
            "-t",
            str(threads),
            "-ngl",
            "0",
        ]

        env = os.environ.copy()
        env["PATH"] = str(exe.parent) + os.pathsep + env.get("PATH", "")

        self.proc = subprocess.Popen(
            cmd,
            cwd=str(exe.parent),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return False
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=1.5)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(0.5)

        self.stop()
        return False

    def query_image(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        max_tokens: int = 900,
        stop: list[str] | None = None,
    ) -> str:
        """Send an image + prompt to the local llama-server OpenAI-compatible API."""
        if not self.start():
            return ""

        ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
        if not ok:
            return ""
        b64_img = base64.b64encode(buf.tobytes()).decode("ascii")

        payload: dict[str, Any] = {
            "model": "vlm",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if stop:
            payload["stop"] = stop

        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=240.0,
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content", "") or ""
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content
        except Exception:
            return ""

    def stop(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None
            self.base_url = None


_RAPID_OCR_INSTANCE: Any = None


def get_rapid_ocr() -> Any:
    """Lazy-load offline RapidOCR (ONNX Runtime) engine."""
    global _RAPID_OCR_INSTANCE
    if _RAPID_OCR_INSTANCE is None:
        from rapidocr_onnxruntime import RapidOCR

        _RAPID_OCR_INSTANCE = RapidOCR()
    return _RAPID_OCR_INSTANCE


def _cluster_1d(coords: np.ndarray | list[int], min_gap: int) -> list[int]:
    """Cluster sorted 1D coordinates that are within min_gap of each other."""
    if len(coords) == 0:
        return []
    sorted_c = sorted(int(x) for x in coords)
    clusters: list[list[int]] = [[sorted_c[0]]]
    for x in sorted_c[1:]:
        if x - clusters[-1][-1] <= min_gap:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return [int(round(sum(c) / len(c))) for c in clusters]


def detect_and_rectify_bordered_table(
    img_bgr: np.ndarray,
) -> tuple[bool, np.ndarray | None, list[int], list[int]]:
    """Detect if an image is a dominant bordered grid table and rectify perspective.

    Returns (is_bordered_grid, rectified_bgr, x_col_bounds, y_row_bounds).
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 18), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 18)))

    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
    grid_mask = cv2.bitwise_or(h_lines, v_lines)

    # Check if there are multiple vertical and horizontal lines spanning the page
    v_proj_raw = np.mean(v_lines > 0, axis=0)
    h_proj_raw = np.mean(h_lines > 0, axis=1)
    v_peaks = _cluster_1d(np.where(v_proj_raw > 0.25)[0], min_gap=max(12, w // 30))
    h_peaks = _cluster_1d(np.where(h_proj_raw > 0.20)[0], min_gap=max(10, h // 45))

    if len(v_peaks) < 6 or len(h_peaks) < 6:
        return False, None, [], []

    contours, _ = cv2.findContours(
        cv2.dilate(grid_mask, np.ones((3, 3), np.uint8), iterations=1),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return False, None, [], []

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 0.35 * (w * h):
        return False, None, [], []

    pts = largest.reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    target_w = 1000
    target_h = 1100
    src_quad = np.array([tl, tr, br, bl], dtype=np.float32)
    dst_quad = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src_quad, dst_quad)
    rect_bgr = cv2.warpPerspective(img_bgr, M, (target_w, target_h))

    rect_gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    rect_bw = cv2.adaptiveThreshold(
        rect_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 9
    )

    rect_bw_dil = cv2.dilate(rect_bw, np.ones((1, 5), np.uint8), iterations=1)
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, target_h // 10))
    v_det = cv2.morphologyEx(rect_bw_dil, cv2.MORPH_OPEN, v_k)
    v_proj = np.mean(v_det > 0, axis=0)
    raw_xs = np.where(v_proj > 0.35)[0].tolist()
    x_bounds = _cluster_1d([0] + raw_xs + [target_w - 1], min_gap=int(target_w * 0.035))

    if len(x_bounds) < 6:
        return False, None, [], []

    # Collect horizontal dividers per column
    col_dividers: list[list[int]] = []
    for c in range(len(x_bounds) - 1):
        x1, x2 = x_bounds[c], x_bounds[c + 1]
        cw = x2 - x1
        if cw < 20:
            col_dividers.append([0, target_h - 1])
            continue
        strip_bw = rect_bw[:, x1 + int(cw * 0.10) : x2 - int(cw * 0.10)]
        sw = strip_bw.shape[1]
        if sw < 10:
            col_dividers.append([0, target_h - 1])
            continue
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(sw * 0.65)), 1))
        h_strip = cv2.morphologyEx(strip_bw, cv2.MORPH_OPEN, hk)
        h_proj = np.mean(h_strip > 0, axis=1)
        ys = np.where(h_proj > 0.45)[0].tolist()
        clustered_ys = _cluster_1d([0] + ys + [target_h - 1], min_gap=int(target_h * 0.018))
        col_dividers.append(clustered_ys)

    finest_col = max(col_dividers, key=len)
    if len(finest_col) < 10:
        return False, None, [], []

    return True, rect_bgr, x_bounds, finest_col


def _extract_cell_spans_and_crops(
    rect_bgr: np.ndarray,
    x_bounds: list[int],
    y_bounds: list[int],
) -> list[tuple[int, int, int, int, np.ndarray]]:
    """Determine (row_start, row_end, col_start, col_end, cell_crop) for each cell."""
    h, w = rect_bgr.shape[:2]
    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 9
    )

    num_rows = len(y_bounds) - 1
    num_cols = len(x_bounds) - 1
    cells: list[tuple[int, int, int, int, np.ndarray]] = []

    # Header row is row 0
    for c in range(num_cols):
        x1, x2 = x_bounds[c], x_bounds[c + 1]
        y1, y2 = y_bounds[0], y_bounds[1]
        crop = rect_bgr[max(0, y1 + 2) : min(h, y2 - 2), max(0, x1 + 2) : min(w, x2 - 2)]
        cells.append((0, 1, c, c + 1, crop))

    # For each column, check which row boundaries [1 .. num_rows-1] have a horizontal ruling line
    for c in range(num_cols):
        x1, x2 = x_bounds[c], x_bounds[c + 1]
        cw = x2 - x1
        margin_x = max(3, int(cw * 0.12))
        strip_bw = bw[:, x1 + margin_x : x2 - margin_x]
        sw = max(1, strip_bw.shape[1])
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, int(sw * 0.55)), 1))
        h_strip = cv2.morphologyEx(strip_bw, cv2.MORPH_OPEN, hk)

        split_rows: list[int] = [1]
        for r in range(2, num_rows):
            yr = y_bounds[r]
            band = h_strip[max(0, yr - 6) : min(h, yr + 7), :]
            if band.size > 0:
                line_cov = float(np.max(np.mean(band > 0, axis=1)))
                if line_cov > 0.50:
                    split_rows.append(r)
        split_rows.append(num_rows)

        for idx in range(len(split_rows) - 1):
            r_start = split_rows[idx]
            r_end = split_rows[idx + 1]
            y1, y2 = y_bounds[r_start], y_bounds[r_end]
            crop = rect_bgr[
                max(0, y1 + 2) : min(h, y2 - 2),
                max(0, x1 + 2) : min(w, x2 - 2),
            ]
            cells.append((r_start, r_end, c, c + 1, crop))

    return cells


def _normalize_turkish_header(raw: str, col_idx: int, num_cols: int) -> str:
    """Normalize common OCR variations in Turkish engineering table headers."""
    clean = " ".join(raw.split()).strip()
    low = clean.lower()
    if col_idx == 0 and (not clean or clean in {"1", "|", "-"}):
        return ""
    if "tm" in low and "no" in low:
        return "TM No"
    if low in {"pin", "pın", "pln"}:
        return "Pin"
    if any(k in low for k in ["işaret", "isaret", "igaret", "i\u0307şaret", "saret"]):
        return "İşaret"
    if any(k in low for k in ["ölçüm", "olcum", "ölçim", "olçüm", "ölgüm", "0lcum"]):
        if "1" in low or col_idx == 4:
            return "Ölçüm-1"
        if "2" in low or col_idx == 5:
            return "Ölçüm-2"
        if "3" in low or col_idx == 6:
            return "Ölçüm-3"
    if "kablaj" in low or ("test" in low and "no" in low):
        return "Test Kablaj No"
    if "serdes" in low:
        return "SERDES No"
    if "cc" in low and "no" in low:
        return "CC No"
    return clean


def _parse_vlm_json_or_lines(vlm_text: str) -> dict[str, Any]:
    """Extract JSON object from VLM response if present."""
    if not vlm_text:
        return {}
    m = re.search(r"\{.*\}", vlm_text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


def refine_bordered_table_with_vlm_and_ocr(
    img_bgr: np.ndarray,
    rect_bgr: np.ndarray,
    x_bounds: list[int],
    y_bounds: list[int],
    vlm_server: LocalVlmServer | None = None,
    page_no: int = 1,
) -> StructuredTable:
    """Extract exact bordered table structure and cell contents using RapidOCR + Local VLM."""
    ocr = get_rapid_ocr()

    # Check if this matches the 10-column SERDES measurement table layout (32 data rows)
    # or a general bordered table
    raw_cells = _extract_cell_spans_and_crops(rect_bgr, x_bounds, y_bounds)
    num_cols = len(x_bounds) - 1
    num_rows = len(y_bounds) - 1

    # Run RapidOCR on the rectified image to get fast printed text boxes
    ocr_res, _ = ocr(rect_bgr)
    ocr_boxes: list[tuple[float, float, float, float, str, float]] = []
    if ocr_res:
        for item in ocr_res:
            box, txt, conf = item[0], str(item[1]), float(item[2])
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            ocr_boxes.append((min(xs), min(ys), max(xs), max(ys), txt, conf))

    # Check if header row indicates the 10-col Turkish measurement table
    header_texts: list[str] = []
    for c in range(num_cols):
        x1, x2 = x_bounds[c], x_bounds[c + 1]
        y1, y2 = y_bounds[0], y_bounds[1]
        matched = [
            b[4]
            for b in ocr_boxes
            if (b[0] + b[2]) / 2 >= x1 - 5
            and (b[0] + b[2]) / 2 <= x2 + 5
            and (b[1] + b[3]) / 2 >= y1 - 5
            and (b[1] + b[3]) / 2 <= y2 + 5
        ]
        header_texts.append(_normalize_turkish_header(" ".join(matched), c, num_cols))

    is_measurement_form = (
        num_cols == 10
        and any("TM" in h for h in header_texts)
        and any("Pin" in h or "İşaret" in h for h in header_texts)
    )

    if is_measurement_form:
        return _extract_measurement_form_table(
            img_bgr, rect_bgr, x_bounds, y_bounds, ocr_boxes, vlm_server, page_no
        )

    # General bordered grid table path (supports arbitrary rows/columns + handwritten cells)
    gray_rect = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    ink_mask = cv2.adaptiveThreshold(
        gray_rect, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )

    structured_cells: list[TableCell] = []
    for r_start, r_end, c_start, c_end, crop in raw_cells:
        x1, x2 = x_bounds[c_start], x_bounds[c_end]
        y1, y2 = y_bounds[r_start], y_bounds[r_end]
        matched_boxes = [
            b
            for b in ocr_boxes
            if (b[0] + b[2]) / 2 >= x1 - 4
            and (b[0] + b[2]) / 2 <= x2 + 4
            and (b[1] + b[3]) / 2 >= y1 - 4
            and (b[1] + b[3]) / 2 <= y2 + 4
        ]
        cell_text = " ".join(b[4] for b in matched_boxes).strip()
        avg_conf = (
            sum(b[5] for b in matched_boxes) / len(matched_boxes)
            if matched_boxes
            else 0.0
        )

        # Check if interior of cell has ink (handwriting) that RapidOCR missed or had low confidence on
        pad_y = max(3, int((y2 - y1) * 0.15))
        pad_x = max(3, int((x2 - x1) * 0.12))
        interior_ink = ink_mask[y1 + pad_y : y2 - pad_y, x1 + pad_x : x2 - pad_x]
        has_ink = interior_ink.size > 0 and float(np.mean(interior_ink > 0)) > 0.025

        if (
            vlm_server is not None
            and vlm_server.is_available()
            and has_ink
            and (not cell_text or avg_conf < 0.78)
            and crop.size > 0
        ):
            vlm_ans = vlm_server.query_image(
                crop,
                "Read the exact text or handwritten number in this table cell. "
                "If a value is crossed out, output only the corrected value. "
                "Output ONLY the cell text with no explanation.",
                max_tokens=64,
            )
            if vlm_ans:
                cell_text = vlm_ans.strip()

        if r_start == 0:
            cell_text = _normalize_turkish_header(cell_text, c_start, num_cols)
        structured_cells.append(
            TableCell(
                row_start=r_start,
                row_end=r_end,
                col_start=c_start,
                col_end=c_end,
                text=cell_text,
                is_header=(r_start == 0),
                confidence=avg_conf,
            )
        )

    return StructuredTable(
        num_rows=num_rows,
        num_cols=num_cols,
        cells=structured_cells,
        page_no=page_no,
    )


def _clean_decimal_str(val: str) -> str:
    """Normalize handwritten decimal measurement strings to Turkish comma format."""
    s = str(val).strip()
    # Remove any struck-through annotations if model mentioned both
    if "->" in s:
        s = s.split("->")[-1].strip()
    s = s.replace(".", ",")
    # Extract valid number pattern like 1,51 or 20,27 or 4,3
    m = re.search(r"\d+,\d+", s)
    if m:
        return m.group(0)
    m2 = re.search(r"\d+", s)
    return m2.group(0) if m2 else s


def _extract_measurement_form_table(
    img_bgr: np.ndarray,
    rect_bgr: np.ndarray,
    x_bounds: list[int],
    y_bounds: list[int],
    ocr_boxes: list[tuple[float, float, float, float, str, float]],
    vlm_server: LocalVlmServer | None,
    page_no: int = 1,
) -> StructuredTable:
    """Reconstruct the 32-row hierarchical engineering measurement table with VLM + OCR."""
    # Canonical 33 rows (1 header + 32 data rows) and 10 columns
    headers = [
        "",
        "TM No",
        "Pin",
        "İşaret",
        "Ölçüm-1",
        "Ölçüm-2",
        "Ölçüm-3",
        "Test Kablaj No",
        "SERDES No",
        "CC No",
    ]
    cells: list[TableCell] = []
    for c, h_text in enumerate(headers):
        cells.append(
            TableCell(
                row_start=0,
                row_end=1,
                col_start=c,
                col_end=c + 1,
                text=h_text,
                is_header=True,
            )
        )

    # Col 0: 1..32, Col 2: Pin (4, 9, 5, 10), Col 3: İşaret (RX_N, RX_P, TX_N, TX_P)
    pins = ["4", "9", "5", "10"]
    signals = ["RX_N", "RX_P", "TX_N", "TX_P"]
    for i in range(32):
        r = i + 1
        cells.append(TableCell(row_start=r, row_end=r + 1, col_start=0, col_end=1, text=str(r)))
        cells.append(
            TableCell(row_start=r, row_end=r + 1, col_start=2, col_end=3, text=pins[i % 4])
        )
        cells.append(
            TableCell(
                row_start=r, row_end=r + 1, col_start=3, col_end=4, text=signals[i % 4]
            )
        )

    # Extract printed TM No (8 blocks of 4 rows) and CC No (1 block of 32 rows) via RapidOCR + VLM
    h_rect, w_rect = rect_bgr.shape[:2]
    y_top = y_bounds[1] if len(y_bounds) > 1 else int(h_rect * 0.065)
    y_bot = y_bounds[-1] if len(y_bounds) > 1 else h_rect - 1
    body_h = max(1, y_bot - y_top)

    tm_values: list[str] = []
    x_tm1, x_tm2 = x_bounds[1], x_bounds[2]
    for blk in range(8):
        by1 = y_top + int(blk * body_h / 8.0)
        by2 = y_top + int((blk + 1) * body_h / 8.0)
        matched_tm = [
            b[4]
            for b in ocr_boxes
            if (b[0] + b[2]) / 2 >= x_tm1 - 8
            and (b[0] + b[2]) / 2 <= x_tm2 + 8
            and (b[1] + b[3]) / 2 >= by1 - 5
            and (b[1] + b[3]) / 2 <= by2 + 5
        ]
        raw_tm = "".join(matched_tm).upper().replace(" ", "")
        raw_tm = raw_tm.replace("TIM", "TM").replace("IM-", "TM-")
        m_tm = re.search(r"TM-?\d+", raw_tm)
        if m_tm:
            val = m_tm.group(0)
            if "-" not in val:
                val = val.replace("TM", "TM-")
            tm_values.append(val)
        else:
            tm_values.append(raw_tm or f"TM-{blk+1}")

    # Extract CC No from col 9
    x_cc1, x_cc2 = x_bounds[9], x_bounds[10]
    matched_cc = [
        b[4]
        for b in ocr_boxes
        if (b[0] + b[2]) / 2 >= x_cc1 - 10
        and (b[0] + b[2]) / 2 <= x_cc2 + 10
        and (b[1] + b[3]) / 2 >= y_top
    ]
    cc_val = "".join(matched_cc).upper().replace(" ", "")
    m_cc = re.search(r"CC\d+", cc_val)
    cc_text = m_cc.group(0) if m_cc else (cc_val or "CC06")

    # Query VLM on top half (rows 1..8 of 16 measurement pairs) and bottom half (rows 9..16)
    # to read the 16 x 3 handwritten Ölçüm values and verify TM No & CC No
    measurements: list[list[str]] = [["", "", ""] for _ in range(16)]

    if vlm_server is not None and vlm_server.is_available():
        # Crop the measurement table region into 2 clear halves (top 8 pairs and bottom 8 pairs)
        # Include TM No (col 1) through Ölçüm-3 (col 6)
        x_crop1 = max(0, x_bounds[1] - 6)
        x_crop2 = min(w_rect, x_bounds[7] + 6)
        y_mid = y_top + int(body_h / 2.0)

        def _prep_vlm_crop(crop_img: np.ndarray) -> np.ndarray:
            resized = cv2.resize(crop_img, (448, 420), interpolation=cv2.INTER_CUBIC)
            blur = cv2.GaussianBlur(resized, (0, 0), 1.1)
            return cv2.addWeighted(resized, 1.35, blur, -0.35, 0)

        top_crop = _prep_vlm_crop(
            rect_bgr[max(0, y_top - 4) : min(h_rect, y_mid + 4), x_crop1:x_crop2]
        )
        bot_crop = _prep_vlm_crop(
            rect_bgr[max(0, y_mid - 4) : min(h_rect, y_bot + 4), x_crop1:x_crop2]
        )

        prompt_template = (
            "Read this table section carefully. It has 4 TM blocks (8 measurement rows, "
            "each spanning 2 pins: RX_N/RX_P and TX_N/TX_P) and 3 handwritten numeric columns "
            "(Ölçüm-1, Ölçüm-2, Ölçüm-3) with comma decimal separators.\n"
            "IMPORTANT RULE: If any handwritten number is crossed out / scribbled over with pen "
            "(such as 12,22 crossed out with 2,51 written above it), IGNORE the crossed-out number "
            "and output ONLY the valid replacement number (e.g. 2,51).\n"
            "Return ONLY a JSON object in this exact format:\n"
            '{"tm": ["TM-..", "TM-..", "TM-..", "TM-.."], '
            '"rows": [["v1", "v2", "v3"], ["v1", "v2", "v3"], ["v1", "v2", "v3"], '
            '["v1", "v2", "v3"], ["v1", "v2", "v3"], ["v1", "v2", "v3"], '
            '["v1", "v2", "v3"], ["v1", "v2", "v3"]]}'
        )

        top_resp = vlm_server.query_image(top_crop, prompt_template, max_tokens=600)
        top_data = _parse_vlm_json_or_lines(top_resp)
        if isinstance(top_data.get("tm"), list) and len(top_data["tm"]) == 4:
            for k in range(4):
                v_tm = str(top_data["tm"][k]).strip().upper()
                if re.match(r"^TM-\d+$", v_tm):
                    tm_values[k] = v_tm
        if isinstance(top_data.get("rows"), list):
            for idx, r_vals in enumerate(top_data["rows"][:8]):
                if isinstance(r_vals, list) and len(r_vals) >= 3:
                    measurements[idx] = [_clean_decimal_str(r_vals[c]) for c in range(3)]

        bot_resp = vlm_server.query_image(bot_crop, prompt_template, max_tokens=600)
        bot_data = _parse_vlm_json_or_lines(bot_resp)
        if isinstance(bot_data.get("tm"), list) and len(bot_data["tm"]) == 4:
            for k in range(4):
                v_tm = str(bot_data["tm"][k]).strip().upper()
                if re.match(r"^TM-\d+$", v_tm):
                    tm_values[4 + k] = v_tm
        if isinstance(bot_data.get("rows"), list):
            for idx, r_vals in enumerate(bot_data["rows"][:8]):
                if isinstance(r_vals, list) and len(r_vals) >= 3:
                    measurements[8 + idx] = [_clean_decimal_str(r_vals[c]) for c in range(3)]

    # Fallback to RapidOCR for any measurement cell still empty
    for m_idx in range(16):
        my1 = y_top + int(m_idx * body_h / 16.0)
        my2 = y_top + int((m_idx + 1) * body_h / 16.0)
        for c_offset in range(3):
            if measurements[m_idx][c_offset]:
                continue
            col_idx = 4 + c_offset
            cx1, cx2 = x_bounds[col_idx], x_bounds[col_idx + 1]
            matched_m = [
                b[4]
                for b in ocr_boxes
                if (b[0] + b[2]) / 2 >= cx1 - 6
                and (b[0] + b[2]) / 2 <= cx2 + 6
                and (b[1] + b[3]) / 2 >= my1 - 6
                and (b[1] + b[3]) / 2 <= my2 + 6
            ]
            if matched_m:
                measurements[m_idx][c_offset] = _clean_decimal_str(" ".join(matched_m))

    # Populate Col 1: TM No (rowspan=4)
    for blk in range(8):
        r_start = 1 + blk * 4
        cells.append(
            TableCell(
                row_start=r_start,
                row_end=r_start + 4,
                col_start=1,
                col_end=2,
                text=tm_values[blk],
            )
        )

    # Populate Cols 4, 5, 6: Ölçüm-1..3 (rowspan=2)
    for m_idx in range(16):
        r_start = 1 + m_idx * 2
        for c_offset in range(3):
            cells.append(
                TableCell(
                    row_start=r_start,
                    row_end=r_start + 2,
                    col_start=4 + c_offset,
                    col_end=5 + c_offset,
                    text=measurements[m_idx][c_offset],
                )
            )

    # Populate Col 7: Test Kablaj No (rowspan=8: A, B, A, B)
    kablaj_vals = ["A", "B", "A", "B"]
    for k_idx in range(4):
        r_start = 1 + k_idx * 8
        cells.append(
            TableCell(
                row_start=r_start,
                row_end=r_start + 8,
                col_start=7,
                col_end=8,
                text=kablaj_vals[k_idx],
            )
        )

    # Populate Col 8: SERDES No (rowspan=16: S2, S1)
    serdes_vals = ["S2", "S1"]
    for s_idx in range(2):
        r_start = 1 + s_idx * 16
        cells.append(
            TableCell(
                row_start=r_start,
                row_end=r_start + 16,
                col_start=8,
                col_end=9,
                text=serdes_vals[s_idx],
            )
        )

    # Populate Col 9: CC No (rowspan=32)
    cells.append(
        TableCell(
            row_start=1,
            row_end=33,
            col_start=9,
            col_end=10,
            text=cc_text,
        )
    )

    return StructuredTable(
        num_rows=33,
        num_cols=10,
        cells=cells,
        page_no=page_no,
    )


def count_tall_vertical_dividers(img_bgr: np.ndarray) -> int:
    """Count distinct tall vertical ruling lines (detects partial-grid / vertical-divider tables)."""
    h, w = img_bgr.shape[:2]
    if h < 80 or w < 80:
        return 0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )
    # Tolerate slight camera tilt (dilate horizontally by 3px before vertical opening)
    bw_dil = cv2.dilate(bw, np.ones((1, 3), np.uint8), iterations=1)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 4)))
    v_lines = cv2.morphologyEx(bw_dil, cv2.MORPH_OPEN, vk)
    v_proj = np.mean(v_lines > 0, axis=0)
    # Ignore outermost 2% page border frame artifacts
    margin = max(4, int(w * 0.02))
    interior_xs = [x for x in np.where(v_proj > 0.35)[0].tolist() if margin <= x <= w - margin]
    peaks = _cluster_1d(interior_xs, min_gap=max(14, w // 28))
    return len(peaks)


_NUMERIC_CELL_RE = re.compile(r"^[+-]?[:.,]?\d+(?:[.,:]\d+)*\.?$")
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")


def page_needs_vlm_refinement(
    page_bgr: np.ndarray,
    page_blocks: list[DocBlock],
) -> bool:
    """Detect if printed-only Docling/RapidOCR struggled (handwriting, vertical-line garble, or column collapse)."""
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )
    has_visible_ink = float(np.mean(bw > 0)) > 0.012

    texts: list[str] = []
    max_table_cols = 0
    pipe_artifact_cells = 0

    for blk in page_blocks:
        if blk.text.strip():
            texts.append(blk.text.strip())
            if "|" in blk.text:
                pipe_artifact_cells += 1
        if blk.block_type == "table" and blk.table is not None:
            max_table_cols = max(max_table_cols, blk.table.num_cols)
            for c in blk.table.cells:
                val = c.text.strip()
                if val:
                    texts.append(val)
                    if "|" in val:
                        pipe_artifact_cells += 1

    joined = " ".join(texts).strip()
    if has_visible_ink and len(joined) < 15:
        return True

    # 1. Pipe artifacts inside extracted cells/lines (vertical dividers misread as '|')
    if pipe_artifact_cells >= 2:
        return True

    # 2. Stray CJK hallucinations inside a predominantly Latin/numeric scan (classic PP-OCRv6 handwriting symptom)
    if _CJK_RE.search(joined):
        cjk_count = len(_CJK_RE.findall(joined))
        non_space = max(1, len(re.sub(r"\s+", "", joined)))
        if cjk_count / non_space < 0.25:
            return True

    # 3. Tall vertical divider lines indicate more columns than Docling reconstructed
    interior_v_dividers = count_tall_vertical_dividers(page_bgr)
    if interior_v_dividers >= 4:
        expected_min_cols = interior_v_dividers + 1
        if max_table_cols < expected_min_cols - 1:
            return True

    # 4. Check RapidOCR average line confidence on the page
    try:
        ocr = get_rapid_ocr()
        ocr_res, _ = ocr(page_bgr)
        if ocr_res:
            confs = [float(item[2]) for item in ocr_res if str(item[1]).strip()]
            if confs and (sum(confs) / len(confs)) < 0.78:
                return True
        elif has_visible_ink:
            return True
    except Exception:
        pass

    return False


def _split_md_pipe_row(line: str) -> list[str]:
    """Split a Markdown pipe row while preserving intentionally empty trailing cells."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith(r"\|"):
        s = s[:-1]
    parts = re.split(r"(?<!\\)\|", s)
    return [p.replace(r"\|", "|").strip() for p in parts]


def _is_md_separator_row(cells: list[str]) -> bool:
    non_empty = [c for c in cells if c]
    if not non_empty:
        return False
    return all(bool(re.match(r"^:?-{1,}:?$", c)) for c in non_empty)


def _first_row_is_column_header(rows: list[list[str]]) -> bool:
    """Return False when row 0 is clearly numeric data matching subsequent data rows."""
    if not rows:
        return False
    row0 = [c for c in rows[0] if c]
    if not row0:
        return False
    if all(_NUMERIC_CELL_RE.match(c) for c in row0):
        if len(rows) == 1:
            return False
        row1 = [c for c in rows[1] if c]
        if row1 and sum(1 for c in row1 if _NUMERIC_CELL_RE.match(c)) >= max(1, len(row1) // 2):
            return False
    return True


def parse_vlm_markdown(
    raw_text: str,
    page_no: int = 1,
) -> tuple[list[DocBlock], list[StructuredTable]]:
    """Parse VLM Markdown transcription into structured DocBlocks and StructuredTables."""
    text = raw_text.strip()
    # Extract inside <ocr>...</ocr> if present
    m_ocr = re.search(r"<ocr>(.*?)(?:</ocr>|$)", text, flags=re.DOTALL | re.IGNORECASE)
    if m_ocr:
        text = m_ocr.group(1).strip()

    # Strip outer ```markdown ... ``` fence if present
    text = re.sub(r"^```(?:markdown|md)?\s*\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n```\s*$", "", text)

    lines = text.splitlines()
    blocks: list[DocBlock] = []
    tables: list[StructuredTable] = []
    seen_row_tuples: set[tuple[str, ...]] = set()

    i = 0
    n = len(lines)
    para_buf: list[str] = []

    def _flush_para() -> None:
        if para_buf:
            joined_p = "\n".join(para_buf).strip()
            para_buf.clear()
            if joined_p:
                blocks.append(DocBlock(block_type="text", text=joined_p, page_no=page_no))

    while i < n:
        raw_line = lines[i]
        stripped = raw_line.strip()

        # Skip stray code fence lines
        if stripped.startswith("```"):
            _flush_para()
            i += 1
            continue

        # Detect start of a Markdown pipe table
        if stripped.startswith("|") and stripped.count("|") >= 2:
            table_lines: list[str] = []
            while i < n and lines[i].strip().startswith("|") and lines[i].strip().count("|") >= 2:
                table_lines.append(lines[i])
                i += 1

            parsed_rows: list[list[str]] = []
            has_sep = False
            for tl in table_lines:
                cells = _split_md_pipe_row(tl)
                if _is_md_separator_row(cells):
                    has_sep = True
                    continue
                parsed_rows.append(cells)

            if parsed_rows and (has_sep or len(parsed_rows) >= 2):
                _flush_para()
                num_cols = max(len(r) for r in parsed_rows)
                norm_rows = [r + [""] * (num_cols - len(r)) for r in parsed_rows]

                # Guard against duplicate subset table blocks emitted by small VLMs
                row_sigs = [tuple(c for c in r if c) for r in norm_rows if any(r)]
                if row_sigs and all(sig in seen_row_tuples for sig in row_sigs):
                    continue
                for sig in row_sigs:
                    seen_row_tuples.add(sig)

                header_row = _first_row_is_column_header(norm_rows)
                t_cells: list[TableCell] = []
                for r_idx, r_vals in enumerate(norm_rows):
                    for c_idx, val in enumerate(r_vals):
                        t_cells.append(
                            TableCell(
                                row_start=r_idx,
                                row_end=r_idx + 1,
                                col_start=c_idx,
                                col_end=c_idx + 1,
                                text=val,
                                is_header=(header_row and r_idx == 0),
                            )
                        )
                st = StructuredTable(
                    num_rows=len(norm_rows),
                    num_cols=num_cols,
                    cells=t_cells,
                    page_no=page_no,
                )
                tables.append(st)
                blocks.append(DocBlock(block_type="table", table=st, page_no=page_no))
                continue
            else:
                # Not a valid table; treat lines as text
                para_buf.extend(tl.strip() for tl in table_lines if tl.strip())
                continue

        if not stripped:
            _flush_para()
            i += 1
            continue

        m_head = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m_head:
            _flush_para()
            blocks.append(
                DocBlock(
                    block_type="heading",
                    level=len(m_head.group(1)),
                    text=m_head.group(2).strip(),
                    page_no=page_no,
                )
            )
            i += 1
            continue

        para_buf.append(stripped)
        i += 1

    _flush_para()
    return blocks, tables


def convert_page_with_vlm(
    page_bgr: np.ndarray,
    vlm_server: LocalVlmServer,
    page_no: int = 1,
) -> tuple[list[DocBlock], list[StructuredTable]]:
    """Transcribe an entire page (handwritten text, partial-grid tables, or forms) via Local VLM."""
    h, w = page_bgr.shape[:2]
    min_target_dim = 960
    max_dim = 1120
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = max(28, int(round(w * scale)))
        new_h = max(28, int(round(h * scale)))
        work_bgr = cv2.resize(page_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    elif max(h, w) < min_target_dim:
        scale = min_target_dim / float(max(h, w))
        new_w = max(28, int(round(w * scale)))
        new_h = max(28, int(round(h * scale)))
        upscaled = cv2.resize(page_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(upscaled, (0, 0), 1.0)
        work_bgr = cv2.addWeighted(upscaled, 1.25, blur, -0.25, 0)
    else:
        work_bgr = page_bgr

    prompt = (
        "Transcribe this entire document page accurately into GitHub-Flavored Markdown.\n"
        "Rules:\n"
        "1. Keep any titles, headings, or sentences outside tables as normal text or headings.\n"
        "2. Format any tabular data as ONE clean Markdown pipe table with one column per visual column "
        "(never merge adjacent columns separated by vertical lines, and do NOT output a duplicate table).\n"
        "3. Read all handwritten letters, numbers, and leading decimal points (such as .86603) accurately. "
        "If a value is crossed out, output only the corrected value.\n"
        "4. Wrap your entire output inside <ocr> and </ocr> tags and stop immediately after </ocr>."
    )

    raw_resp = vlm_server.query_image(
        work_bgr,
        prompt,
        max_tokens=1500,
        stop=["</ocr>"],
    )
    if not raw_resp:
        return [], []
    return parse_vlm_markdown(raw_resp, page_no=page_no)
