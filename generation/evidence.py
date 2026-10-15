"""整理单次问答的证据编号、原文位置与可核对的来源列表。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from langchain_core.documents import Document

from retrieval.hybrid_retriever import _doc_key


@dataclass(frozen=True)
class AnswerEvidence:
    citation_id: str
    chunk_id: str
    document_id: str
    source: str
    locations: list[dict]
    text: str


@dataclass(frozen=True)
class CitationAudit:
    cited_ids: list[str]
    unknown_ids: list[str]
    has_citations: bool
    nonstandard_ids: list[str] = field(default_factory=list)


def build_evidence(documents: list[Document]) -> list[AnswerEvidence]:
    """按本次检索顺序编号，相同切块只出现一次，保留旧资料的章节信息。"""
    evidence = []
    seen: set[str] = set()
    for doc in documents:
        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        metadata = doc.metadata
        raw = metadata.get("source_locations_json", "[]")
        try:
            locations = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError) as exc:
            raise ValueError(f"回答证据来源位置损坏：{key}") from exc
        if not isinstance(locations, list) or any(not isinstance(item, dict) for item in locations):
            raise ValueError(f"回答证据来源位置格式错误：{key}")
        legacy = {field: metadata[field] for field in ("chapter", "article_no") if metadata.get(field)}
        if not locations and legacy:
            locations = [legacy]
        evidence.append(AnswerEvidence(f"S{len(evidence) + 1}", key,
                                       str(metadata.get("document_id", "")),
                                       str(metadata.get("source") or "来源未命名"),
                                       locations, doc.page_content))
    return evidence


def evidence_json(evidence: list[AnswerEvidence]) -> str:
    """作为用户消息中的资料数据传入，不把文档内容升级为系统指令。"""
    return json.dumps([asdict(item) for item in evidence], ensure_ascii=False)


def audit_citations(answer: str, evidence: list[AnswerEvidence]) -> CitationAudit:
    """只校验编号是否存在，不把编号有效等同于结论得到证据支持。"""
    # 只兼容完整且成对的三种括号；不猜裸编号、混搭括号或省略的引用。
    matches = list(re.finditer(r"\[(S\d+)\]|\((S\d+)\)|（(S\d+)）", answer))
    cited = list(dict.fromkeys(next(value for value in match.groups() if value) for match in matches))
    nonstandard = list(dict.fromkeys(match.group(2) or match.group(3) for match in matches if not match.group(1)))
    known = {item.citation_id for item in evidence}
    return CitationAudit(cited, [key for key in cited if key not in known], bool(cited), nonstandard)


def location_label(locations: list[dict]) -> str:
    labels = []
    for location in locations:
        parts = []
        for key, title in (("page_number", "页"), ("slide_number", "幻灯片"),
                           ("sheet_name", "工作表"), ("cell_range", "单元格"),
                           ("paragraph_index", "段落"), ("table_index", "表格"),
                           ("row_index", "表格行"), ("chapter", "章节"), ("article_no", "条款")):
            if location.get(key) is not None:
                parts.append(f"{title} {location[key]}")
        if location.get("line_start") is not None:
            start = location["line_start"]
            end = location.get("line_end")
            parts.append(f"行 {start}–{end}" if end is not None and end != start else f"行 {start}")
        if not parts and location.get("parser_locator"):
            parts.append(str(location["parser_locator"]))
        if parts:
            labels.append("，".join(parts))
    return "；".join(dict.fromkeys(labels)) or "未提供页码或位置"


def sources_suffix(answer: str, evidence: list[AnswerEvidence]) -> str:
    """程序根据实际证据追加来源，不接受模型编造的文件路径和引用编号。"""
    audit = audit_citations(answer, evidence)
    by_id = {item.citation_id: item for item in evidence}
    lines = []
    for key in audit.cited_ids:
        if key in by_id:
            item = by_id[key]
            # 文件名按纯文本展示，避免资料元数据被解释为链接或HTML。
            source = item.source.replace("`", "'").replace("\n", " ").replace("\r", " ")
            position = location_label(item.locations).replace("`", "'").replace("\n", " ").replace("\r", " ")
            lines.append(f"- [{key}] `{source}`；`{position}`")
    suffix = "\n\n来源：\n\n" + "\n".join(lines) if lines else ""
    if audit.unknown_ids:
        suffix += "\n\n引用检查：" + "、".join(f"[{key}]" for key in audit.unknown_ids) + "不在本次检索资料中，无法核对这些引用。"
    if audit.nonstandard_ids:
        suffix += "\n\n引用格式提示：检测到非标准括号编号，仅为本次确实存在的编号关联来源；原回答未改写，编号匹配不代表内容已得到证据支持。"
    if evidence and not audit.has_citations:
        suffix += "\n\n引用检查：本回答未标注来源编号，尚未完成引用核对。"
    return suffix
