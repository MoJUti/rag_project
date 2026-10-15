"""验证逐条复核记录后汇总提示词对比，不从引用编号推断答案正确。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_node7_prompt_ab import digest
from scripts.run_node6_rerank_eval import save_cache


def validate_review(row: dict, review: dict, known_ids: set[str]) -> None:
    """复核必须绑定原始答案；遗漏、拼错评分、引用不存在均阻止汇总。"""
    answer = row["raw_answer"]
    if review.get("answer_sha256") != hashlib.sha256(answer.encode()).hexdigest():
        raise ValueError("复核记录与原始答案哈希不符")
    for field in ("correctness", "completeness", "boundary"):
        value = review[field]
        if value is not None and (type(value) is not int or value not in {0, 1, 2}):
            raise ValueError(f"复核评分无效：{field}")
    for claim in review["claims"]:
        if not claim["text"] or claim["text"] not in answer:
            raise ValueError("复核断言不是原始答案中的文字")
        if claim["support"] not in {"supported", "unsupported", "contradicted", "uncertain"}:
            raise ValueError("复核断言的支持状态无效")
        if claim["support"] == "supported" and not set(claim["cited_ids"]) <= known_ids:
            raise ValueError("未知证据编号不能标记为支持断言")
        if type(claim["requires_citation"]) is not bool:
            raise ValueError("断言是否需要引用必须是布尔值")
        if any(f"[{key}]" not in answer for key in claim["cited_ids"]):
            raise ValueError("复核记录声称引用了答案未使用的编号")


def summarize(report: dict, reviews: dict) -> dict:
    """保留各维度分母和组别；无复核或分母为零时不生成虚假百分比。"""
    if reviews["experiment_sha256"] != digest(report["identity"]):
        raise ValueError("复核文件不属于本次提示词对比")
    if set(reviews["rows"]) != set(report["rows"]):
        raise ValueError("复核记录必须覆盖且仅覆盖本次所有答案")
    fixtures = {row["question_id"]: row for row in report["identity"]["fixtures"]}
    expected = {f"{version}/{key}" for version in ("baseline", "candidate") for key in fixtures}
    if set(report["rows"]) != expected:
        raise ValueError("两版回答尚未完整，不能汇总完整实验")
    summary = {"experiment_sha256": digest(report["identity"]), "reviewer": reviews["reviewer"], "versions": {}}
    for version in ("baseline", "candidate"):
        rows = [(key, row) for key, row in report["rows"].items() if key.startswith(version + "/")]
        collected = []
        for key, row in rows:
            if row["status"] != "success":
                continue
            question_id = key.split("/", 1)[1]
            review = reviews["rows"][key]
            validate_review(row, review, {entry["citation_id"] for entry in fixtures[question_id]["evidence"]})
            collected.append((question_id, row, review))
        metrics = {"attempted": len(rows), "success": len(collected), "groups": {}}
        groups = {entry["group"] for entry in fixtures.values()}
        for group in sorted(groups):
            selected = [(row, review) for key, row, review in collected if fixtures[key]["group"] == group]
            claims = [claim for _, review in selected for claim in review["claims"]]
            required = [claim for claim in claims if claim["requires_citation"]]
            cited = [claim for claim in required if claim["cited_ids"]]
            supported = [claim for claim in cited if claim["support"] == "supported"]
            counts = {"answers": len(selected), "required_claims": len(required), "cited_claims": len(cited),
                      "supported_cited_claims": len(supported),
                      "citation_precision": len(supported) / len(cited) if cited else None,
                      "citation_coverage": len(supported) / len(required) if required else None,
                      "critical_error_answers": sum(bool(review["critical_errors"]) for _, review in selected)}
            for field in ("correctness", "completeness", "boundary"):
                values = [review[field] for _, review in selected if review[field] is not None]
                counts[field] = {"sum": sum(values), "count": len(values),
                                 "mean": statistics.fmean(values) if values else None}
            metrics["groups"][group] = counts
        usage = [row["model_info"].get("token_usage", {}) for _, row in rows]
        metrics["usage"] = {field: sum(item.get(field, 0) for item in usage)
                            for field in ("prompt_tokens", "completion_tokens", "total_tokens")}
        metrics["cached_input_tokens"] = sum(item.get("prompt_tokens_details", {}).get("cached_tokens", 0) for item in usage)
        metrics["latency_mean_ms"] = statistics.fmean(row["latency_ms"] for _, row in rows)
        summary["versions"][version] = metrics
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("reviews", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
    result = summarize(report, reviews)
    output = args.report.with_name(args.report.stem + "_summary.json")
    save_cache(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report={output}")
