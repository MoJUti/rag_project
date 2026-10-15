"""离线核对最小证据片段与人工问题记录，不覆盖旧指标，也不自动给答案打分。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.document_model import stable_document_id
from scripts.evaluate_embedded_retrieval import load_chunk_catalog
from scripts.run_chunking_benchmark import read_jsonl
from scripts.run_node6_rerank_eval import save_cache
from scripts.run_node7_prompt_ab import digest


def normalized(text: str) -> str:
    """只统一全半角、空白与Markdown强调；保留数字、单位、符号和表格列界。"""
    value = re.sub(r"[`*_]", "", unicodedata.normalize("NFKC", text).casefold())
    # 先保护相邻数值或英文词的分隔，避免去空白后把两列金额粘成一个数字。
    value = re.sub(r"(?<=[a-z0-9.])\s+(?=[a-z0-9.])", "\u200b", value)
    return re.sub(r"\s", "", value)


def contains_quote(text: str, quote: str) -> bool:
    """匹配连续原文，避免把8000当成800、把0.40当成0.400。"""
    value = normalized(quote)
    if not value:
        raise ValueError("证据原文不能为空")
    left = r"(?<![a-z0-9.])" if value[0] in "0123456789abcdefghijklmnopqrstuvwxyz" else ""
    right = r"(?![a-z0-9.])" if value[-1] in "0123456789abcdefghijklmnopqrstuvwxyz" else ""
    return re.search(left + re.escape(value) + right, normalized(text)) is not None


def match_requirement(requirement: dict, documents: list[dict], mapping: dict[str, str]) -> list[dict]:
    """实体、年份或表头限制必须与原文在同一切块，不能跨无关行拼凑数字。"""
    if not requirement.get("id") or not requirement.get("document_ids") or not requirement.get("alternatives"):
        raise ValueError("最小证据要求缺少ID、来源或原文")
    try:
        allowed = {mapping[key] for key in requirement["document_ids"]}
    except KeyError as exc:
        raise ValueError(f"最小证据来源不在清单中：{exc}") from exc
    for alternative in requirement["alternatives"]:
        if not normalized(alternative.get("quote", "")):
            raise ValueError("证据原文不能为空")
        if not isinstance(alternative.get("context", []), list) or any(
            not isinstance(text, str) or not normalized(text) for text in alternative.get("context", [])
        ):
            raise ValueError("证据上下文限制必须是非空原文列表")
    matches = []
    for document in documents:
        if document["document_id"] not in allowed:
            continue
        for alternative in requirement["alternatives"]:
            if all(contains_quote(document["text"], text)
                   for text in [alternative["quote"], *alternative.get("context", [])]):
                matches.append({key: document[key] for key in ("chunk_id", "citation_id", "source", "locations")
                                if key in document} | {"matched_quote": alternative["quote"]})
                break
    return matches


def audit(report: dict, specification: dict, catalog: dict, mapping: dict) -> dict:
    """所有判断绑定旧实验和实际答案；诊断子集不冒充全量修正召回率。"""
    if specification["experiment_sha256"] != digest(report["identity"]):
        raise ValueError("诊断附表不属于本次实验")
    if specification["report_content_sha256"] != digest(report):
        raise ValueError("诊断附表与报告内容指纹不符")
    if digest(catalog) != report["identity"]["catalog_sha256"]:
        raise ValueError("诊断语料与原实验切块指纹不符")
    corpus = [{"chunk_id": key, **row} for key, row in catalog.items()]
    result = {"identity": {"experiment_sha256": digest(report["identity"]),
                           "specification_sha256": digest(specification),
                           "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "reviewer": specification["reviewer"],
              "review_scope": specification.get("review_scope", "问题样本的事实与来源诊断，非穷尽断言评分"),
              "evidence_rows": {}, "semantic_reviews": {}}
    for key, case in specification["evidence_cases"].items():
        if key not in report["rows"]:
            raise ValueError(f"诊断题号不在实验中：{key}")
        original = report["rows"][key]
        if original["status"] != "success":
            raise ValueError(f"诊断题尚未生成成功：{key}")
        requirements = case["requirements"]
        if not requirements or len({row["id"] for row in requirements}) != len(requirements):
            raise ValueError(f"证据要求为空或ID重复：{key}")
        allowed = set(original["question"].get("relevant_document_ids", []))
        if any(set(row["document_ids"]) - allowed for row in requirements):
            raise ValueError(f"诊断证据超出原题相关来源：{key}")
        for evidence in original["evidence"]:
            frozen = catalog.get(evidence["chunk_id"])
            if not frozen or frozen != {"document_id": evidence["document_id"], "text": evidence["text"]}:
                raise ValueError(f"实际上下文与冻结切块不符：{key}")
        atoms = [{"requirement_id": row["id"],
                  "context_matches": match_requirement(row, original["evidence"], mapping),
                  "corpus_matches": match_requirement(row, corpus, mapping)} for row in requirements]
        complete_context = all(row["context_matches"] for row in atoms)
        complete_corpus = all(row["corpus_matches"] for row in atoms)
        diagnosis = ("minimal_spans_present" if complete_context else "retrieval_gap" if complete_corpus
                     else "upstream_or_annotation_gap")
        result["evidence_rows"][key] = {"diagnosis": diagnosis, "scope_note": case["scope_note"],
            "old_metrics": original["retrieval_metrics"], "requirements": atoms,
            "answer_sha256": hashlib.sha256(original["raw_answer"].encode()).hexdigest()}
    for key, review in specification.get("semantic_reviews", {}).items():
        original = report["rows"].get(key)
        if not original or original.get("status") != "success":
            raise ValueError(f"语义复核题号缺少已完成答案：{key}")
        if review["answer_sha256"] != hashlib.sha256(original["raw_answer"].encode()).hexdigest():
            raise ValueError(f"语义复核与答案哈希不符：{key}")
        if review.get("finding") not in {"citation_format_failure", "supported_abstention", "grounding_failure",
                                         "annotation_ambiguity", "needs_scope_review", "supported_answer",
                                         "partial_answer", "unanswered_with_missing_evidence", "upstream_answer_mismatch",
                                         "scope_or_unit_uncertain", "boundary_omission", "minor_expression_issue"}:
            raise ValueError(f"语义复核结论标签无效：{key}")
        if not review.get("notes") or not review.get("answer_excerpt") or review["answer_excerpt"] not in original["raw_answer"]:
            raise ValueError(f"语义复核缺少说明或原答案摘录：{key}")
        known = {row["citation_id"]: row for row in original["evidence"]}
        for support in review.get("source_checks", []):
            if support["citation_id"] not in known or not contains_quote(known[support["citation_id"]]["text"], support["quote"]):
                raise ValueError(f"语义复核引用的原文不存在：{key}")
        result["semantic_reviews"][key] = review
    diagnoses = [row["diagnosis"] for row in result["evidence_rows"].values()]
    result["summary"] = {"experiment_questions": len(report["rows"]), "evidence_audited": len(diagnoses),
                         "diagnosis_counts": {name: diagnoses.count(name) for name in sorted(set(diagnoses))},
                         "semantic_reviewed": len(result["semantic_reviews"]),
                         "semantic_unreviewed": len(report["rows"]) - len(result["semantic_reviews"]),
                         "findings": {name: sum(row["finding"] == name for row in result["semantic_reviews"].values())
                                      for name in sorted({row["finding"] for row in result["semantic_reviews"].values()})},
                         "overall_answer_accuracy": None, "overall_corrected_recall": None,
                         "note": "事后选择的诊断子集；原文匹配是定位辅助，不等于语义正确或全量评测"}
    return result


def add_reviews(report: dict, specification: dict, additional: dict) -> dict:
    """先绑定完整报告，再为人工填写的结论附上答案哈希；不自动判断答案对错。"""
    for field, expected in (("experiment_sha256", digest(report["identity"])),
                            ("report_content_sha256", digest(report))):
        if additional.get(field) != expected or specification.get(field) != expected:
            raise ValueError(f"补充复核文件与报告指纹不符：{field}")
    combined = dict(specification["semantic_reviews"])
    for key, review in additional["semantic_reviews"].items():
        if key in combined or key not in report["rows"]:
            raise ValueError(f"补充复核题号重复或不在报告中：{key}")
        if "answer_sha256" in review:
            raise ValueError(f"补充输入的答案哈希应由已绑定报告生成：{key}")
        combined[key] = {**review, "answer_sha256": hashlib.sha256(report["rows"][key]["raw_answer"].encode()).hexdigest()}
    if set(combined) != set(report["rows"]):
        raise ValueError("补充复核后仍未覆盖全部答案，不得作为全量复核运行")
    return {**specification, "semantic_reviews": combined, "additional_review_sha256": digest(additional),
            "reviewer": additional["reviewer"], "review_scope": additional["review_scope"]}


def main(report_path: Path, specification_path: Path, additional_path: Path | None = None) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    if additional_path is not None:
        additional = json.loads(additional_path.read_text(encoding="utf-8"))
        specification = add_reviews(report, specification, additional)
    dataset = ROOT / "data/eval/zh_public_minibench_v0.1"
    manifest = read_jsonl(dataset / "manifest.jsonl")
    mapping = {row["item_id"]: stable_document_id(row["sha256"]) for row in manifest}
    catalog = load_chunk_catalog(ROOT / "storage/evaluations/zh_public_minibench_v0.1/chunks/structure_aware_v1")
    result = audit(report, specification, catalog, mapping)
    output = report_path.with_name(f"{report_path.stem}_audit_{digest(result['identity'])[:12]}.json")
    save_cache(output, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"report={output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("specification", type=Path)
    parser.add_argument("--additional-reviews", type=Path, help="补充剩余答案的问题级人工复核，与已有记录合并且要求全量覆盖")
    arguments = parser.parse_args()
    main(arguments.report, arguments.specification, arguments.additional_reviews)
