"""GLM定点回放必须保持真实十块证据与原模型条件。"""

import copy
import unittest
from unittest.mock import patch

from scripts.replay_node7_glm_fixes import build_cases


class GlmFixReplayTest(unittest.TestCase):
    def original(self) -> dict:
        evidence = [{"citation_id": f"S{i + 1}", "chunk_id": f"chunk-{i}", "document_id": "doc",
                     "source": "测试.md", "locations": [], "text": f"原始片段{i}"} for i in range(10)]
        return {"identity": {"provider": "zhipu", "retrieval": {"hybrid_final_k": 10}},
                "rows": {"case": {"status": "success", "question": {"query": "真实问题", "gold_answer": "不能注入的标答"},
                                  "evidence": evidence, "raw_answer": "历史回答"}}}

    def test_preserves_ten_sources_and_repeat_count(self) -> None:
        original = self.original()
        with patch("scripts.replay_node7_glm_fixes.DEFINITIONS", {"case": 3}):
            cases = build_cases(original)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["repeats"], 3)
        self.assertEqual(cases[0]["evidence"], original["rows"]["case"]["evidence"])

    def test_wrong_provider_or_budget_is_rejected(self) -> None:
        for field, value in (("provider", "qwen"), ("retrieval", {"hybrid_final_k": 6})):
            original = self.original()
            original["identity"][field] = value
            with self.assertRaisesRegex(ValueError, "只能回放GLM"):
                build_cases(original)

    def test_missing_failed_or_truncated_context_is_rejected(self) -> None:
        original = self.original()
        for mode in ("missing", "failed", "truncated", "renumbered"):
            changed = copy.deepcopy(original)
            if mode == "missing":
                changed["rows"] = {}
            elif mode == "failed":
                changed["rows"]["case"]["status"] = "failed"
            elif mode == "truncated":
                changed["rows"]["case"]["evidence"].pop()
            else:
                changed["rows"]["case"]["evidence"][0]["citation_id"] = "S99"
            with patch("scripts.replay_node7_glm_fixes.DEFINITIONS", {"case": 1}):
                with self.assertRaises(ValueError):
                    build_cases(changed)


if __name__ == "__main__":
    unittest.main()
