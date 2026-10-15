"""固定证据的提示词对比不能串版本、串会话或把替身成绩当质量评分。"""

import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from generation import prompts
from scripts.run_node7_prompt_ab import build_cases, digest, experiment_identity, load_comparison_prompts, run_answer
from test.test_grounded_answers import FakeVectorService, make_document
from generation.rag_service import RagService
from core import config
from langchain_core.chat_history import InMemoryChatMessageHistory


class PromptABTest(unittest.TestCase):
    def test_history_supplies_baseline_and_current_file_is_the_only_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "history.json"
            report.write_text(json.dumps({"identity": {"system_prompts": {"baseline": "旧正文"}}}), encoding="utf-8")
            systems = load_comparison_prompts(report)
            self.assertEqual(systems["baseline"], "旧正文")
            self.assertEqual(systems["candidate"], prompts.SYSTEM_PROMPT)
            report.write_text('{}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "读取历史基线提示词失败"):
                load_comparison_prompts(report)
            report.write_text(json.dumps({"identity": {"system_prompts": {"baseline": " "}}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "为空或类型错误"):
                load_comparison_prompts(report)

    def test_markdown_examples_remain_static_system_data(self) -> None:
        messages = prompts.build_answer_prompt().invoke({
            "context": "本次唯一证据", "input": "本次问题", "history": [],
        }).to_messages()
        self.assertTrue(messages[0].content.startswith("# 中文知识库问答"))
        self.assertEqual(messages[0].content.count("```json"), 4)
        self.assertIn('"citation_id":"S1"', messages[0].content)
        self.assertNotIn("本次唯一证据", messages[0].content)
        self.assertIn("本次唯一证据", messages[1].content)
        self.assertFalse((Path(__file__).resolve().parents[1] / "prompts/generation/knowledge_answer_v2.md").exists())

    def test_explicit_candidate_does_not_change_application_default(self) -> None:
        original = prompts.SYSTEM_PROMPT
        candidate = prompts.build_answer_prompt("候选规则")
        self.assertEqual(candidate.invoke({"context": "资料", "history": [], "input": "问题"}).to_messages()[0].content,
                         "候选规则")
        self.assertEqual(prompts.SYSTEM_PROMPT, original)
        self.assertNotEqual(prompts.answer_prompt_sha256("候选规则"), prompts.answer_prompt_sha256())

    def test_fixed_cases_keep_real_rankings_and_synthetic_labels(self) -> None:
        doc = make_document()
        spec = {"cases": [{"question_id": "real", "group": "regression"},
                          {"question_id": "fake", "group": "synthetic_boundary", "query": "问题",
                           "gold_answer": "答案", "documents": [{"source": "虚构", "text": "测试"}]}]}
        cases = build_cases(spec, {"real": {"query": "问题", "gold_answer": "答案"}}, {"real": ["a"]}, {"a": doc})
        self.assertEqual(cases[0]["documents"], [doc])
        self.assertEqual(cases[1]["evidence"][0]["document_id"], "synthetic-fake-0")
        spec["cases"].append(spec["cases"][0])
        with self.assertRaisesRegex(ValueError, "题号重复"):
            build_cases(spec, {"real": {}}, {"real": ["a"]}, {"a": doc})

    def test_identity_changes_for_rubric_candidate_and_evidence(self) -> None:
        spec = {"max_tokens": 800}
        systems = {"baseline": "基线", "candidate": "候选"}
        cases = [{"question_id": "a", "group": "regression", "question": {"query": "问题"}, "evidence": []}]
        before = digest(experiment_identity(spec, "规则", systems, cases))
        self.assertNotEqual(before, digest(experiment_identity(spec, "新规则", systems, cases)))
        self.assertNotEqual(before, digest(experiment_identity(spec, "规则", {**systems, "candidate": "新候选"}, cases)))
        cases[0]["evidence"] = [{"text": "新证据"}]
        self.assertNotEqual(before, digest(experiment_identity(spec, "规则", systems, cases)))

    def test_answer_run_records_evidence_boundary_and_never_assigns_semantic_score(self) -> None:
        docs = [make_document()]
        histories = {}
        with patch.object(config, "evidence_expansion", "none"):
            rag = RagService(vector_service=FakeVectorService(docs), chat_model=FakeListChatModel(responses=["47[S1]"]),
                             history_factory=lambda key: histories.setdefault(key, InMemoryChatMessageHistory()))
        case = {"documents": docs, "question": {"query": "问题", "scope_source_path": "documents/md/missing.md"}}
        row = run_answer(rag, case, "isolated")
        self.assertEqual(row["status"], "success")
        self.assertFalse(row["scope_document_present"])
        self.assertEqual(row["review_status"], "awaiting_source_based_review")
        self.assertNotIn("accuracy", row)
        self.assertEqual(len(histories["isolated"].messages), 2)
