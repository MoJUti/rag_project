"""隔离会话比较多轮记忆行为，并观察真实知识库指代追问；不修改生产策略。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, message_to_dict, messages_from_dict
from core import config
from core.prompt_loader import load_prompt
from generation.prompts import answer_prompt_sha256
from memory.file_history import FileChatMessageHistory, ContextChatMessageHistory
from memory.history_store import _build_light_model, _summarize_messages
from memory.session_context import estimate_tokens, fingerprint
from scripts.benchmark_memory_answers import timed_answer
from scripts.run_node6_rerank_eval import save_cache

FIXTURE = ROOT / "data/eval/session_memory_v1/cases.json"


def turns(rows: list) -> list:
    return [message for user, assistant in rows for message in (HumanMessage(content=user), AIMessage(content=assistant))]


def padding(target_tokens: int, label: str) -> list:
    """无业务新事实的流程文本，仅用于真实预算下触发压缩，不作为问答证据。"""
    result, total = [], 0
    while total < target_tokens:
        pair = turns([[f"{label}流程举例{len(result)//2+1}：\n" +
                       "目录检查和版本核对是流程举例，不改变项目预算、日期、审批码、供应商或禁止事项。\n" * 25,
                       "这只是流程举例，不产生任何新的业务事实。"]])
        result.extend(pair)
        total += estimate_tokens(pair)
    return result


def exact_score(text: str, probe: dict) -> dict:
    """null答案也须确实输出该字段，不能把缺字段或无效JSON算作正确遗忘。"""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"format_valid": False, "correct": False}
    valid = isinstance(value, dict) and set(value) == {probe["field"]}
    actual = value.get(probe["field"]) if valid else None
    expected = probe["expected"]
    return {"format_valid": valid, "actual": actual,
            "correct": valid and type(actual) is type(expected) and actual == expected}


def prepare_contexts(report: dict, output: Path, fixture: dict) -> None:
    """两次真实读写检查点，第二次将更正和遗忘也推进增量摘要。"""
    (ROOT / "tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
        def summarize(messages: list, existing: str) -> Any:
            started = time.perf_counter()
            result = _summarize_messages(messages, existing)
            report.setdefault("summary_calls", []).append({"existing_summary": existing,
                "input_messages": [message_to_dict(m) for m in messages], "result": asdict(result),
                "elapsed_seconds": time.perf_counter() - started})
            # 先存付费输出，再进行上下文状态保存。
            save_cache(output, report)
            return result

        archive = FileChatMessageHistory("behavior_fixture", directory, summarize)
        first = turns(fixture["initial_turns"]) + padding(34000, "第一段")
        archive.add_messages(first)
        archive.context_messages()
        report["first_checkpoint"] = {"state": archive.context_state, "message_count": len(archive.messages)}
        save_cache(output, report)
        if archive.context_state["status"] != "ready":
            raise RuntimeError("第一个记忆检查点降级，停止付费测试")
        archive.add_messages(turns(fixture["update_turns"]) + padding(34000, "第二段") + turns(fixture["recent_turns"]))
        context = archive.context_messages()
        state = archive.context_state
        if state["status"] != "ready" or len(report.get("summary_calls", [])) != 2:
            raise RuntimeError("第二个记忆检查点未正常完成两次增量摘要，停止测试")
        full = archive.messages
        covered = state["covered_messages"]
        updated_user = fixture["update_turns"][0][0]
        if not any(m.content == updated_user for m in full[:covered]):
            raise ValueError("更正未进入摘要覆盖范围，不能声称验证压缩后更正")
        report["contexts"] = {"full_history": [message_to_dict(m) for m in full],
            "recent_window": [message_to_dict(m) for m in full[-6:]],
            "summary_32k_4k": [message_to_dict(m) for m in context]}
        report["final_state"] = state
        report["full_message_count"] = len(full)
        report["context_estimated_tokens"] = {name: estimate_tokens(messages_from_dict(rows))
                                                for name, rows in report["contexts"].items()}
        save_cache(output, report)


def run_memory(report: dict, output: Path, fixture: dict) -> None:
    prompt = load_prompt("memory/behavior_answer_probe.md")
    failures = 0
    for probe in fixture["probes"]:
        for mode, context in report["contexts"].items():
            key = f"{probe['id']}:{mode}"
            if key in report["rows"]:
                continue
            data = {"history": [{"role": m["type"], "content": m["data"]["content"]} for m in context],
                    "question": probe["question"], "field": probe["field"]}
            result = timed_answer(_build_light_model(), [SystemMessage(content=prompt),
                HumanMessage(content=json.dumps(data, ensure_ascii=False))])
            result["score"] = exact_score(result["text"], probe)
            result.update({"probe_id": probe["id"], "mode": mode})
            report["rows"][key] = result
            save_cache(output, report)
            print(json.dumps({"case": key, "status": result["status"], "score": result["score"],
                              "total_seconds": result["total_seconds"]}, ensure_ascii=False), flush=True)
            failures = failures + 1 if result["status"] == "failed" else 0
            if failures >= 3:
                raise RuntimeError("记忆问答连续三次失败，已保存并停止")


def run_retrieval(report: dict, output: Path, fixture: dict) -> None:
    """同一历史比较实际省略式追问与人工明确对象对照；不自动改写生产查询。"""
    from infra.vector_store import VectorStoreService
    from infra.vector_stores.chroma import ChromaVectorStoreAdapter
    from ingestion.storage import StorageLayout
    from ingestion.document_model import stable_document_id
    from scripts.run_node7_end_to_end import ObservedRagService, load_questions_and_vectors, run_case
    from scripts.evaluate_embedded_retrieval import load_chunk_catalog
    from scripts.run_node6_retrieval_baseline import CachedQuestionEmbeddings
    from scripts.run_chunking_benchmark import read_jsonl
    from retrieval.compatible_embeddings import CompatibleEmbeddings

    evaluation = StorageLayout.from_env(ROOT).evaluation("zh_public_minibench_v0.1")
    bundle = ROOT / "data/eval/zh_public_minibench_v0.1"
    questions, vectors = load_questions_and_vectors(bundle, evaluation)
    questions = {q["question_id"]: q for q in questions}
    catalog = load_chunk_catalog(evaluation.chunks / "structure_aware_v1")
    mapping = {r["item_id"]: stable_document_id(r["sha256"]) for r in read_jsonl(bundle / "manifest.jsonl")}
    missing = [p["followup"] for p in fixture["retrieval_pairs"] if p["followup"] not in report.get("followup_vectors", {})]
    if missing:
        embedding = CompatibleEmbeddings.from_env(base_url=config.embedding_base_url, model=config.embedding_model_name,
                         dimensions=config.embedding_dimensions, batch_size=config.embedding_batch_size)
        report.setdefault("followup_vectors", {}).update(zip(missing, embedding.embed_documents(missing)))
        save_cache(output, report)
    vectors.update(report["followup_vectors"])
    vector = VectorStoreService(adapter=ChromaVectorStoreAdapter(CachedQuestionEmbeddings(vectors)))
    actual = {d.metadata["chunk_id"]: {"document_id": d.metadata["document_id"], "text": d.page_content}
              for d in vector.get_all_documents()}
    if actual != catalog:
        raise ValueError("应用Chroma与冻结切块不同，停止检索对照，不改库")
    digest = fingerprint(catalog)
    if report.get("catalog_sha256", digest) != digest:
        raise ValueError("冻结切块已改变，不能复用本次对照")
    report["catalog_sha256"] = digest
    (ROOT / "tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
        for pair in fixture["retrieval_pairs"]:
            original = questions[pair["source_question_id"]]
            for mode, query in (("followup", pair["followup"]), ("explicit_control", original["query"])):
                key = f"{pair['id']}:{mode}"
                if key in report.setdefault("retrieval_rows", {}):
                    continue
                session_id = key.replace(":", "_")
                archive = FileChatMessageHistory(session_id, directory)
                archive.add_messages(turns(pair["history"]))
                rag = ObservedRagService(vector_service=vector, chat_model=_build_light_model().bind(max_tokens=800),
                    history_factory=lambda _, a=archive: ContextChatMessageHistory(a))
                question = {**original, "query": query, "question_id": key}
                result = run_case(rag, question, catalog, mapping, session_id)
                result["history"] = pair["history"]
                # 标准答案字符串仅为诊断；语义、单位与引用支持关系仍须原文复核。
                result["literal_gold_present"] = original["gold_answer"] in result.get("raw_answer", "")
                report["retrieval_rows"][key] = result
                save_cache(output, report)
                print(json.dumps({"case": key, "status": result["status"],
                    "retrieval_metrics": result.get("retrieval_metrics"),
                    "literal_gold_present": result["literal_gold_present"]}, ensure_ascii=False), flush=True)
                if result["status"] != "success":
                    raise RuntimeError("知识库对照调用失败，已保存并停止，不自动重试")


def main(live: bool, retrieval: bool) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if (config.memory_history_max_tokens, config.memory_summary_max_tokens, config.memory_compression_trigger) != (32768, 4096, "tokens"):
        raise ValueError("本轮必须使用已确认的32K/4K Token策略")
    keys = ("collection_name", "chroma_mode", "persist_directory", "chroma_host", "chroma_port",
            "retrieval_strategy", "bm25_tokenizer", "hybrid_vector_k", "hybrid_bm25_k", "hybrid_final_k",
            "hybrid_rrf_k", "rerank_strategy", "rerank_candidate_k", "evidence_expansion",
            "embedding_base_url", "embedding_model_name", "embedding_dimensions")
    identity = {"protocol": "memory-behaviors-v1", "fixture": fixture,
                "model": config.zhipu_chat_model, "base_url": config.zhipu_base_url, "reasoning_effort": "low",
                "budget": 32768, "summary_cap": 4096, "output_cap": 800,
                "answer_prompt": load_prompt("memory/behavior_answer_probe.md"),
                "summary_prompt": (ROOT / "prompts/memory/conversation_summary.md").read_text(encoding="utf-8"),
                "rag_prompt_sha256": answer_prompt_sha256(), "retrieval_config": {k: getattr(config, k) for k in keys},
                "code_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
                    "scripts/evaluate_memory_behaviors.py", "scripts/benchmark_memory_answers.py",
                    "memory/session_context.py", "memory/history_store.py", "memory/file_history.py",
                    "generation/rag_service.py", "retrieval/pipeline.py", "retrieval/hybrid_retriever.py")}}
    output = ROOT / f"storage/evaluations/session_memory_v1/reports/behaviors_{fingerprint(identity)[:12]}.json"
    report = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"identity": identity, "rows": {}}
    if report["identity"] != identity:
        raise ValueError("行为评测缓存身份不匹配")
    print(f"report={output}; live={live}; planned_summary_calls=2; planned_memory_answers=21; retrieval_answers=4", flush=True)
    if not live:
        return
    if "contexts" not in report:
        if report.get("summary_calls"):
            raise RuntimeError("存在未完成的付费摘要检查点，需人工确认后恢复；不重复调用")
        prepare_contexts(report, output, fixture)
    run_memory(report, output, fixture)
    if retrieval:
        run_retrieval(report, output, fixture)
    print(f"report={output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--retrieval", action="store_true", help="同时运行两道真实知识库追问与明确对象对照")
    args = parser.parse_args()
    main(args.live, args.retrieval)
