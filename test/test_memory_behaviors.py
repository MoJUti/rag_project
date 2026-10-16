"""中文多轮事实核验边界与真实预算压缩检查点。"""

import json
import unittest

from scripts.evaluate_memory_behaviors import FIXTURE, exact_score, padding, turns
from memory.session_context import build_context, estimate_tokens


class MemoryBehaviorTests(unittest.TestCase):
    def test_forgetting_requires_explicit_null_field(self) -> None:
        probe = {"field": "deadline", "expected": None}
        self.assertTrue(exact_score('{"deadline":null}', probe)["correct"])
        for text in ("{}", "[]", '{"deadline":null,"other":1}', "不是JSON"):
            self.assertFalse(exact_score(text, probe)["correct"])

    def test_score_distinguishes_boolean_integer_and_string(self) -> None:
        probe = {"field": "budget", "expected": 1}
        self.assertFalse(exact_score('{"budget":true}', probe)["correct"])
        self.assertFalse(exact_score('{"budget":"1"}', probe)["correct"])
        self.assertTrue(exact_score('{"budget":1}', probe)["correct"])

    def test_two_checkpoints_cover_correction_and_do_not_destroy_originals(self) -> None:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        source = turns(fixture["initial_turns"]) + padding(34000, "第一段")
        calls = []

        def summarize(messages: list, previous: str) -> str:
            calls.append((messages, previous))
            return "替身摘要，仅验证覆盖游标。"

        _, state = build_context(source, {}, summarize)
        source += turns(fixture["update_turns"]) + padding(34000, "第二段") + turns(fixture["recent_turns"])
        context, state = build_context(source, state, summarize)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][1], "替身摘要，仅验证覆盖游标。")
        self.assertTrue(any(m.content == fixture["update_turns"][0][0] for m in calls[1][0]))
        self.assertLessEqual(estimate_tokens(context), 32768)
        self.assertTrue(any(m.content == fixture["initial_turns"][0][0] for m in source))
