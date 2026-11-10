"""Exporters for HTML (native rowspan/colspan), CSV (fill-down UTF-8 BOM), and Markdown (GFM)."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import html
from pathlib import Path


@dataclass
class TableCell:
    """Represents a single logical table cell with row/column spans."""

    row_start: int
    row_end: int  # exclusive
    col_start: int
    col_end: int  # exclusive
    text: str
    is_header: bool = False
    confidence: float = 1.0

    @property
    def rowspan(self) -> int:
        return max(1, self.row_end - self.row_start)

    @property
    def colspan(self) -> int:
        return max(1, self.col_end - self.col_start)


@dataclass
class StructuredTable:
    """Represents a reconstructed table with exact span geometry."""

    num_rows: int
    num_cols: int
    cells: list[TableCell] = field(default_factory=list)
    caption: str = ""
    page_no: int = 1

    def to_unrolled_grid(self, fill_down: bool = True) -> list[list[str]]:
        """Convert spanned cells into a 2D matrix of strings.

        When fill_down=True, vertically/horizontally merged cell values are repeated
        across every spanned row/column so each row is a complete record.
        """
        grid: list[list[str]] = [
            ["" for _ in range(self.num_cols)] for _ in range(self.num_rows)
        ]
        for cell in self.cells:
            clean_text = " ".join(cell.text.replace("\r", " ").split())
            r_end = min(self.num_rows, cell.row_end)
            c_end = min(self.num_cols, cell.col_end)
            if fill_down:
                for r in range(cell.row_start, r_end):
                    for c in range(cell.col_start, c_end):
                        grid[r][c] = clean_text
            else:
                if 0 <= cell.row_start < self.num_rows and 0 <= cell.col_start < self.num_cols:
                    grid[cell.row_start][cell.col_start] = clean_text
        return grid

    def to_html(self) -> str:
        """Render HTML <table> preserving native rowspan and colspan attributes."""
        # Map top-left (row_start, col_start) -> TableCell
        cell_map: dict[tuple[int, int], TableCell] = {}
        covered: set[tuple[int, int]] = set()

        for cell in self.cells:
            key = (cell.row_start, cell.col_start)
            cell_map[key] = cell
            for r in range(cell.row_start, min(self.num_rows, cell.row_end)):
                for c in range(cell.col_start, min(self.num_cols, cell.col_end)):
                    if (r, c) != key:
                        covered.add((r, c))

        lines: list[str] = ['<table class="ocr-table">']
        if self.caption:
            lines.append(f"  <caption>{html.escape(self.caption)}</caption>")

        has_header = any(c.is_header for c in self.cells)
        if has_header and self.num_rows > 0:
            lines.append("  <thead>")
            lines.append(self._render_html_row(0, cell_map, covered, force_th=True))
            lines.append("  </thead>")
            start_body_row = 1
        else:
            start_body_row = 0

        if start_body_row < self.num_rows:
            lines.append("  <tbody>")
            for r in range(start_body_row, self.num_rows):
                lines.append(self._render_html_row(r, cell_map, covered, force_th=False))
            lines.append("  </tbody>")

        lines.append("</table>")
        return "\n".join(lines)

    def _render_html_row(
        self,
        r: int,
        cell_map: dict[tuple[int, int], TableCell],
        covered: set[tuple[int, int]],
        force_th: bool = False,
    ) -> str:
        row_parts: list[str] = ["    <tr>"]
        for c in range(self.num_cols):
            if (r, c) in covered:
                continue
            cell = cell_map.get((r, c))
            if cell is None:
                tag = "th" if force_th else "td"
                row_parts.append(f"      <{tag}></{tag}>")
                continue

            tag = "th" if (force_th or cell.is_header) else "td"
            attrs: list[str] = []
            if cell.rowspan > 1:
                attrs.append(f'rowspan="{cell.rowspan}"')
            if cell.colspan > 1:
                attrs.append(f'colspan="{cell.colspan}"')
            attr_str = (" " + " ".join(attrs)) if attrs else ""
            escaped = html.escape(cell.text.strip()).replace("\n", "<br>")
            row_parts.append(f"      <{tag}{attr_str}>{escaped}</{tag}>")
        row_parts.append("    </tr>")
        return "\n".join(row_parts)

    def to_markdown(self, fill_down: bool = True) -> str:
        """Render GitHub-Flavored Markdown pipe table."""
        if self.num_rows == 0 or self.num_cols == 0:
            return ""
        grid = self.to_unrolled_grid(fill_down=fill_down)

        # Compute column widths for readable alignment
        col_widths = [3] * self.num_cols
        for row in grid:
            for c, val in enumerate(row):
                escaped_val = val.replace("|", "\\|")
                col_widths[c] = max(col_widths[c], len(escaped_val))

        def fmt_row(row_vals: list[str]) -> str:
            parts = []
            for c, val in enumerate(row_vals):
                escaped_val = val.replace("|", "\\|")
                parts.append(escaped_val.ljust(col_widths[c]))
            return "| " + " | ".join(parts) + " |"

        md_lines: list[str] = []
        if self.caption:
            md_lines.append(f"### {self.caption}\n")

        has_header = any(c.is_header for c in self.cells)
        sep_parts = ["-" * w for w in col_widths]
        if has_header:
            md_lines.append(fmt_row(grid[0]))
            md_lines.append("| " + " | ".join(sep_parts) + " |")
            for r in range(1, self.num_rows):
                md_lines.append(fmt_row(grid[r]))
        else:
            md_lines.append(fmt_row([""] * self.num_cols))
            md_lines.append("| " + " | ".join(sep_parts) + " |")
            for r in range(0, self.num_rows):
                md_lines.append(fmt_row(grid[r]))

        return "\n".join(md_lines)


@dataclass
class DocBlock:
    """A document block in reading order (text, heading, or table)."""

    block_type: str  # "heading", "text", "table"
    text: str = ""
    level: int = 2
    table: StructuredTable | None = None
    page_no: int = 1


@dataclass
class DocumentConversionResult:
    """Complete result of converting a single image or PDF document."""

    source_path: Path
    blocks: list[DocBlock] = field(default_factory=list)
    tables: list[StructuredTable] = field(default_factory=list)
    docling_markdown: str | None = None
    docling_html: str | None = None


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    :root {{
      --border-color: #334155;
      --header-bg: #d9a74a;
      --header-text: #1e293b;
      --row-even: #f8fafc;
      --text-color: #0f172a;
    }}
    body {{
      font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      color: var(--text-color);
      background: #ffffff;
      margin: 2rem auto;
      max-width: 1200px;
      padding: 0 1.5rem;
      line-height: 1.5;
    }}
    h1, h2, h3 {{
      color: #1e293b;
      margin-top: 1.5rem;
      margin-bottom: 0.75rem;
    }}
    .doc-header {{
      border-bottom: 2px solid #cbd5e1;
      padding-bottom: 0.5rem;
      margin-bottom: 1.5rem;
    }}
    .doc-paragraph {{
      margin: 0.6rem 0;
      white-space: pre-wrap;
    }}
    .ocr-table {{
      border-collapse: collapse;
      width: 100%;
      margin: 1.5rem 0;
      font-size: 0.95rem;
      background: #fff;
    }}
    .ocr-table th,
    .ocr-table td {{
      border: 1.5px solid var(--border-color);
      padding: 6px 10px;
      text-align: center;
      vertical-align: middle;
    }}
    .ocr-table thead th {{
      background-color: var(--header-bg);
      color: var(--header-text);
      font-weight: 700;
    }}
    .ocr-table tbody tr:nth-child(even) {{
      background-color: var(--row-even);
    }}
    @media print {{
      body {{
        margin: 0;
        padding: 0.5cm;
      }}
    }}
  </style>
</head>
<body>
  <div class="doc-header">
    <h1>{title}</h1>
  </div>
{body}
</body>
</html>
"""


def export_html(result: DocumentConversionResult, output_path: Path) -> Path:
    """Export document result to a standalone HTML file with native rowspan/colspan."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body_parts: list[str] = []

    if result.blocks:
        for block in result.blocks:
            if block.block_type == "heading":
                lvl = min(6, max(1, block.level))
                body_parts.append(f"  <h{lvl}>{html.escape(block.text)}</h{lvl}>")
            elif block.block_type == "text" and block.text.strip():
                body_parts.append(
                    f'  <p class="doc-paragraph">{html.escape(block.text)}</p>'
                )
            elif block.block_type == "table" and block.table is not None:
                body_parts.append(block.table.to_html())
    elif result.tables:
        for tbl in result.tables:
            body_parts.append(tbl.to_html())
    elif result.docling_html:
        output_path.write_text(result.docling_html, encoding="utf-8")
        return output_path

    full_html = HTML_TEMPLATE.format(
        title=html.escape(result.source_path.name),
        body="\n".join(body_parts),
    )
    output_path.write_text(full_html, encoding="utf-8")
    return output_path


def export_markdown(
    result: DocumentConversionResult,
    output_path: Path,
    fill_down: bool = True,
) -> Path:
    """Export document result to a GitHub-Flavored Markdown file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    md_parts: list[str] = []

    if result.blocks:
        for block in result.blocks:
            if block.block_type == "heading":
                prefix = "#" * min(6, max(1, block.level))
                md_parts.append(f"{prefix} {block.text.strip()}\n")
            elif block.block_type == "text" and block.text.strip():
                md_parts.append(f"{block.text.strip()}\n")
            elif block.block_type == "table" and block.table is not None:
                md_parts.append(block.table.to_markdown(fill_down=fill_down) + "\n")
    elif result.tables:
        for tbl in result.tables:
            md_parts.append(tbl.to_markdown(fill_down=fill_down) + "\n")
    elif result.docling_markdown:
        md_parts.append(result.docling_markdown)

    content = "\n".join(md_parts).strip() + "\n"
    output_path.write_text(content, encoding="utf-8")
    return output_path


def export_csv(
    result: DocumentConversionResult,
    output_path: Path,
    delimiter: str = ",",
    fill_down: bool = True,
) -> list[Path]:
    """Export extracted tables to CSV (UTF-8 with BOM for Excel compatibility)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []

    if not result.tables:
        # If no structured tables were detected, write key-value/text lines into CSV
        with output_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Page", "Type", "Content"])
            for block in result.blocks:
                if block.text.strip():
                    writer.writerow([block.page_no, block.block_type, block.text.strip()])
        written_files.append(output_path)
        return written_files

    # Check if all tables share the same column schema (e.g., multi-page continuation)
    first_grid = result.tables[0].to_unrolled_grid(fill_down=fill_down)
    same_schema = len(result.tables) > 1 and all(
        t.num_cols == result.tables[0].num_cols for t in result.tables
    )

    if len(result.tables) == 1 or same_schema:
        with output_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
            for idx, tbl in enumerate(result.tables):
                grid = tbl.to_unrolled_grid(fill_down=fill_down)
                if not grid:
                    continue
                # Skip duplicate header row on continuation pages if identical
                start_r = 1 if (idx > 0 and grid[0] == first_grid[0]) else 0
                for row in grid[start_r:]:
                    writer.writerow(row)
        written_files.append(output_path)
    else:
        # Write primary table to output_path and additional tables to _table_N.csv
        for idx, tbl in enumerate(result.tables):
            target = (
                output_path
                if idx == 0
                else output_path.with_name(f"{output_path.stem}_table_{idx + 1}.csv")
            )
            grid = tbl.to_unrolled_grid(fill_down=fill_down)
            with target.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
                for row in grid:
                    writer.writerow(row)
            written_files.append(target)

    return written_files
