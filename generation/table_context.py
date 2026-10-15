"""实验性证据呈现：只对行列明确的Markdown表格展开表头，不猜合并单元格。"""

from __future__ import annotations

from dataclasses import asdict
import json
import re

from generation.evidence import AnswerEvidence


def cells(line: str) -> list[str] | None:
    """仅识别完整管道行；转义分隔符、HTML和非标准排版保持原样。"""
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|") or "\\|" in line or "<" in line:
        return None
    return [part.strip() for part in stripped[1:-1].split("|")]


def expand_tables(text: str) -> str:
    """把可验证的矩形表按位置展开；空值、单位、正负号和重复列均不推断。"""
    lines = text.splitlines(keepends=True)
    output = []
    index = 0
    while index < len(lines):
        header = cells(lines[index])
        separator = cells(lines[index + 1]) if index + 1 < len(lines) else None
        if not header or not separator or len(header) != len(separator) or not all(
            re.fullmatch(r":?-{3,}:?", part) for part in separator
        ):
            output.append(lines[index])
            index += 1
            continue
        end = index + 2
        rows = []
        valid = True
        while end < len(lines) and lines[end].lstrip().startswith("|"):
            row = cells(lines[end])
            if row is None or len(row) != len(header):
                valid = False
            rows.append(row)
            end += 1
        if not valid or not rows:
            output.extend(lines[index:end])
        else:
            # 用位置和原列名组成数组，避免空表头或重名字段在JSON字典里被覆盖。
            named = [{"原表行序": number, "单元格": [
                {"列序": column, "表头": name, "值": value}
                for column, (name, value) in enumerate(zip(header, row), 1)]}
                for number, row in enumerate(rows, 1)]
            output.append("表格按原表头展开（空字符串保持空值，不补单位）：\n")
            output.extend(json.dumps(row, ensure_ascii=False) + "\n" for row in named)
        index = end
    return "".join(output)


def named_cell_evidence_json(evidence: list[AnswerEvidence]) -> str:
    """只改变送入模型的呈现，原始证据对象仍用于引用与历史回放。"""
    return json.dumps([{**asdict(item), "text": expand_tables(item.text)} for item in evidence], ensure_ascii=False)
