"""显式测试历史消歧插件，冻结模型决策后走同一正式回答链，不切换默认策略。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import message_to_dict
from core import config
from core.prompt_loader import load_prompt
from generation.prompts import answer_prompt_sha256
from infra.vector_store import VectorStoreService
from infra.vector_stores.chroma import ChromaVectorStoreAdapter
from ingestion.storage import StorageLayout
from ingestion.document_model import stable_document_id
from memory.file_history import FileChatMessageHistory, ContextChatMessageHistory
from memory.history_store import _build_light_model
from memory.session_context import fingerprint
from retrieval.compatible_embeddings import CompatibleEmbeddings
from retrieval.query_resolution import HistoryQueryResolver, QueryResolution
from scripts.evaluate_embedded_retrieval import load_chunk_catalog
from scripts.evaluate_memory_behaviors import turns
from scripts.run_chunking_benchmark import read_jsonl
from scripts.run_node6_retrieval_baseline import CachedQuestionEmbeddings
from scripts.run_node6_rerank_eval import save_cache
from scripts.run_node7_end_to_end import ObservedRagService, load_questions_and_vectors, run_case


class RecordedResolver:
    """评测复用本轮已付费决策，严格绑定输入；生产插件不缓存用户历史。"""

    def __init__(self, decision: QueryResolution, history: list) -> None:
        self.decision = decision
        self.history = [message_to_dict(m) for m in history]

    def resolve(self, query: str, history: list) -> QueryResolution:
        if query != self.decision.original_query or [message_to_dict(m) for m in history] != self.history:
            raise ValueError("冻结消歧决策与正式链实际历史不一致")
        return self.decision


class PreparedRag(ObservedRagService):
    """只在评测实例中保存本题准备结果，包含澄清短路的实际阶段记录。"""

    def _prepare_answer(self, value: dict) -> dict:
        result = super()._prepare_answer(value)
        self.prepared = result
        return result


def main(live: bool) -> None:
    if (config.vector_store_backend, config.embedding_model_name, config.embedding_dimensions, config.hybrid_final_k) != (
        "chroma", "text-embedding-v4", 2048, 10):
        raise ValueError("本轮要求现有Chroma、text-embedding-v4/2048、最终10块，不自动改配置")
    fixture = json.loads((ROOT / "data/eval/session_memory_v1/query_resolution_cases.json").read_text(encoding="utf-8"))
    evaluation = StorageLayout.from_env(ROOT).evaluation("zh_public_minibench_v0.1")
    bundle = ROOT / "data/eval/zh_public_minibench_v0.1"
    questions, vectors = load_questions_and_vectors(bundle, evaluation)
    questions = {q["question_id"]: q for q in questions}
    catalog = load_chunk_catalog(evaluation.chunks / "structure_aware_v1")
    mapping = {r["item_id"]: stable_document_id(r["sha256"]) for r in read_jsonl(bundle / "manifest.jsonl")}
    identity = {"protocol": "history-query-resolution-v1", "fixture": fixture,
                "model": config.zhipu_chat_model, "base_url": config.zhipu_base_url, "reasoning_effort": "low",
                "query_prompt": load_prompt("retrieval/history_query.md"), "answer_prompt_sha256": answer_prompt_sha256(),
                "catalog_sha256": fingerprint(catalog), "max_tokens": 800,
                "sources": {c["source_question_id"]: questions[c["source_question_id"]] for c in fixture["cases"] if "source_question_id" in c},
                "retrieval_config": {key: getattr(config, key) for key in (
                    "collection_name", "chroma_mode", "persist_directory", "chroma_host", "chroma_port",
                    "retrieval_strategy", "hybrid_vector_k", "hybrid_bm25_k", "hybrid_final_k", "hybrid_rrf_k",
                    "bm25_tokenizer", "rerank_strategy", "rerank_candidate_k", "evidence_expansion", "embedding_base_url")},
                "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                    "scripts/evaluate_query_resolution.py", "retrieval/query_resolution.py", "retrieval/pipeline.py",
                    "generation/rag_service.py", "generation/model_pool.py")}}
    output = ROOT / f"storage/evaluations/session_memory_v1/reports/query_resolution_{fingerprint(identity)[:12]}.json"
    report = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {
        "identity": identity, "decisions": {}, "vectors": {}, "rows": {}}
    if report["identity"] != identity:
        raise ValueError("消歧评测缓存身份不一致")
    print(f"report={output}; live={live}; planned_decisions=8; planned_answer_calls=6", flush=True)
    if not live:
        return
    resolver = HistoryQueryResolver()
    for case in fixture["cases"]:
        key = case["id"]
        if key in report["decisions"]:
            if report["decisions"][key]["status"] != "completed":
                raise RuntimeError(f"{key}已有失败决策，保留记录，不自动重试")
            continue
        try:
            decision = resolver.resolve(case["query"], turns(case["history"]))
        except (RuntimeError, ValueError) as exc:
            report["decisions"][key] = {"status": "failed", "error": str(exc)}
            save_cache(output, report)
            raise
        report["decisions"][key] = {"status": "completed", "decision": asdict(decision),
            "expected_action": case["expected_action"], "action_correct": decision.action == case["expected_action"]}
        save_cache(output, report)
        print(json.dumps({"case": key, "action": decision.action, "query": decision.query,
            "clarification": decision.clarification, "expected_action": case["expected_action"]}, ensure_ascii=False), flush=True)
    pending = list(dict.fromkeys(d["decision"]["query"] for d in report["decisions"].values()
                        if d["decision"]["action"] != "clarify" and d["decision"]["query"] not in vectors
                        and d["decision"]["query"] not in report["vectors"]))
    if pending:
        embedding = CompatibleEmbeddings.from_env(base_url=config.embedding_base_url, model=config.embedding_model_name,
                         dimensions=config.embedding_dimensions, batch_size=config.embedding_batch_size)
        try:
            report["vectors"].update(zip(pending, embedding.embed_documents(pending)))
        finally:
            embedding.close()
        save_cache(output, report)
    vectors.update(report["vectors"])
    vector = VectorStoreService(adapter=ChromaVectorStoreAdapter(CachedQuestionEmbeddings(vectors)))
    actual = {d.metadata["chunk_id"]: {"document_id": d.metadata["document_id"], "text": d.page_content}
              for d in vector.get_all_documents()}
    if actual != catalog:
        raise ValueError("应用库与冻结切块不同，不运行或修改数据库")
    (ROOT / "tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
        for case in fixture["cases"]:
            key = case["id"]
            if key in report["rows"]:
                continue
            decision = QueryResolution(**report["decisions"][key]["decision"])
            history = turns(case["history"])
            archive = FileChatMessageHistory(key, directory)
            archive.add_messages(history)
            rag = PreparedRag(vector_service=vector, chat_model=_build_light_model().bind(max_tokens=800),
                query_resolver=RecordedResolver(decision, history), history_factory=lambda _, a=archive: ContextChatMessageHistory(a))
            source = questions.get(case.get("source_question_id"))
            if source is None and decision.action != "clarify":
                report["rows"][key] = {"status": "semantic_failure", "reason": "应澄清的无唯一对象题却执行查询", "decision": asdict(decision)}
                save_cache(output, report)
                continue
            question = {**(source or {"answerable": False}), "query": case["query"], "question_id": key}
            row = run_case(rag, question, catalog, mapping, key)
            if hasattr(rag, "prepared"):
                row["decision"] = rag.prepared["query_resolution"]
                row["full_trace"] = [asdict(stage) for stage in rag.prepared["retrieval_trace"]]
            row["resolution_plus_answer_ms"] = decision.elapsed_ms + row["elapsed_ms"]
            row["retrieval_called"] = rag.last_retrieval is not None
            row["literal_gold_present"] = source["gold_answer"] in row.get("raw_answer", "") if source else None
            report["rows"][key] = row
            save_cache(output, report)
            print(json.dumps({"case": key, "status": row["status"], "retrieval_called": row["retrieval_called"],
                "metrics": row.get("retrieval_metrics"), "literal_gold_present": row["literal_gold_present"]}, ensure_ascii=False), flush=True)
            if row["status"] != "success":
                raise RuntimeError("正式链对照失败，保留记录并停止，不自动重试")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    main(parser.parse_args().live)
