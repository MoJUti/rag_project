"""回答修复回放保留原始证据，不在失败题里悄悄补标准资料。"""

import unittest

from scripts.replay_node7_answer_fixes import NoLiveVectorStore, build_replays, restore_documents


class AnswerFixReplayTest(unittest.TestCase):
    def evidence(self, chunk_id: str) -> dict:
        return {"citation_id": "S1", "chunk_id": chunk_id, "document_id": "doc1", "source": "测试.md",
                "locations": [{"line_start": 3}], "text": "不把${input}当模板变量"}

    def test_restore_preserves_snapshot_and_rejects_changed_numbering(self) -> None:
        evidence = [self.evidence("original")]
        docs = restore_documents(evidence)
        self.assertEqual(docs[0].page_content, evidence[0]["text"])
        self.assertEqual(docs[0].metadata["chunk_id"], "original")
        with self.assertRaisesRegex(ValueError, "编号或来源"):
            restore_documents([{**evidence[0], "citation_id": "S2"}])
        with self.assertRaisesRegex(ValueError, "编号或来源"):
            restore_documents([evidence[0], evidence[0]])
        with self.assertRaisesRegex(RuntimeError, "不允许重新召回"):
            NoLiveVectorStore().get_vector_docs("问题", 6)

    def test_repeats_and_complete_context_are_separate_from_original_missing_context(self) -> None:
        keys = ("accept-v1-r01", "accept-v1-x04", "accept-v1-a04", "accept-v1-a06", "accept-v1-a11",
                "wb-37-02-01", "accept-v1-x01", "accept-v1-r02", "accept-v1-a05")
        original = {"identity": {"system_prompt": "旧规则"}, "rows": {key: {
            "status": "success", "question": {"query": "问题"}, "raw_answer": "历史答案",
            "evidence": [self.evidence("original")],
        } for key in keys}}
        comparison = {"identity": {"system_prompts": {"candidate": "旧规则"}, "fixtures": [{
            "question_id": "accept-v1-x04", "question": {"query": "问题"}, "evidence": [self.evidence("full")],
        }]}, "rows": {"candidate/accept-v1-x04": {"raw_answer": "完整对照"}}}
        cases = build_replays(original, comparison)
        self.assertEqual(len(cases), 10)
        self.assertEqual(sum(case["repeats"] for case in cases), 14)
        missing = next(case for case in cases if case["case_id"] == "accept-v1-x04")
        self.assertEqual(missing["evidence"][0]["chunk_id"], "original")
        self.assertEqual(cases[-1]["evidence"][0]["chunk_id"], "full")
        comparison["identity"]["system_prompts"]["candidate"] = "另一版本"
        with self.assertRaisesRegex(ValueError, "历史提示词不同"):
            build_replays(original, comparison)
