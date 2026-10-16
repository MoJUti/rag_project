"""记忆问答评测的评分和流式计时回归测试。"""

import json
import unittest
from types import SimpleNamespace

from scripts.benchmark_memory_answers import EXPECTED, score_answer, timed_answer


class MemoryAnswerBenchmarkTests(unittest.TestCase):
    def test_exact_fields_and_unknown(self) -> None:
        self.assertTrue(score_answer(json.dumps(EXPECTED))["complete_correct"])
        value = json.loads(json.dumps(EXPECTED))
        value["P001"]["budget_yuan"] = True
        value["P024"]["delivery_date"] = None
        result = score_answer(json.dumps(value))
        self.assertEqual(result["correct_fields"], 7)
        self.assertEqual(result["details"]["P001.budget_yuan"], "wrong")
        self.assertEqual(result["details"]["P024.delivery_date"], "unknown")

    def test_invalid_json(self) -> None:
        for text in ("```json\n{}\n```", "[]", "不是JSON"):
            self.assertFalse(score_answer(text)["format_valid"])

    def test_first_nonempty_chunk_and_usage(self) -> None:
        chunks = [SimpleNamespace(content="", response_metadata={}),
                  SimpleNamespace(content="{}", response_metadata={"usage": {"total_tokens": 3}})]
        model = SimpleNamespace(stream=lambda *a, **k: iter(chunks))
        ticks = iter([10., 12., 15.])
        result = timed_answer(model, [], lambda: next(ticks))
        self.assertEqual(result["first_text_seconds"], 2.)
        self.assertEqual(result["total_seconds"], 5.)
        self.assertEqual(result["metadata"]["usage"]["total_tokens"], 3)

    def test_partial_failure_is_not_success(self) -> None:
        def chunks(*args, **kwargs):
            yield SimpleNamespace(content="{", response_metadata={})
            raise RuntimeError("测试中断")
        result = timed_answer(SimpleNamespace(stream=chunks), [])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["text"], "{")
