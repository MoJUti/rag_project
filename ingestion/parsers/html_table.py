"""将 MinerU 输出的 HTML 表格转换为稳定的单元格模型。"""

from __future__ import annotations

from html.parser import HTMLParser

from ingestion.document_model import TableCell, TableData


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int, bool]]] = []
        self._row: list[tuple[str, int, int, bool]] | None = None
        self._cell: list[str] | None = None
        self._rowspan = 1
        self._colspan = 1
        self._header = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
            self._rowspan = max(1, _safe_int(attributes.get("rowspan"), 1))
            self._colspan = max(1, _safe_int(attributes.get("colspan"), 1))
            self._header = tag == "th"
        elif tag in {"br", "p", "div", "li"} and self._cell is not None:
            self._cell.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            text = " ".join("".join(self._cell).split())
            self._row.append((text, self._rowspan, self._colspan, self._header))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def strip_html(value: str) -> str:
    parser = _TextParser()
    parser.feed(value)
    return " ".join("".join(parser.parts).split())


def parse_html_table(html: str, caption: str | None = None) -> TableData:
    parser = _TableParser()
    parser.feed(html)
    occupied: set[tuple[int, int]] = set()
    cells: list[TableCell] = []
    max_row = 0
    max_column = 0

    for row_index, source_row in enumerate(parser.rows):
        column = 0
        for text, row_span, column_span, is_header in source_row:
            while (row_index, column) in occupied:
                column += 1
            cells.append(
                TableCell(
                    row=row_index,
                    column=column,
                    text=text,
                    row_span=row_span,
                    column_span=column_span,
                    is_header=is_header,
                    raw_value=text,
                )
            )
            for row in range(row_index, row_index + row_span):
                for col in range(column, column + column_span):
                    occupied.add((row, col))
            max_row = max(max_row, row_index + row_span)
            max_column = max(max_column, column + column_span)
            column += column_span

    return TableData(
        row_count=max_row,
        column_count=max_column,
        cells=cells,
        caption=caption,
    )


def _safe_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


__all__ = ["parse_html_table", "strip_html"]
