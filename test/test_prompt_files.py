"""提示词加载边界、数据替换、迁移快照与版本指纹的回归测试。"""

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core import config, prompt_loader
from generation import prompts
from ingestion.parsers.qwen_fallback import build_review_prompt
from ingestion.parsers.review_models import AnomalyType
from memory.history_store import _build_summary_prompt


class PromptLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.root_patch = patch.object(prompt_loader, "PROMPT_ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        prompt_loader.load_prompt.cache_clear()
        self.addCleanup(prompt_loader.load_prompt.cache_clear)

    def write(self, text: str) -> None:
        (self.root / "sample.md").write_text(text, encoding="utf-8")

    def test_utf8_content_json_braces_and_input_are_not_reinterpreted(self) -> None:
        self.write('规则：{"text": "中文"}\n资料：${context}\n价格：$$10\n')
        actual = prompt_loader.render_prompt("sample.md", {"context": '${unknown} {input}\n原文'})
        self.assertEqual(actual, '规则：{"text": "中文"}\n资料：${unknown} {input}\n原文\n价格：$10')

    def test_missing_file_has_explicit_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "读取失败.*missing.md"):
            prompt_loader.load_prompt("missing.md")

    def test_empty_and_invalid_utf8_have_explicit_errors(self) -> None:
        self.write(" \n")
        with self.assertRaisesRegex(ValueError, "文件为空"):
            prompt_loader.load_prompt("sample.md")
        (self.root / "sample.md").write_bytes(b"\xff")
        with self.assertRaisesRegex(RuntimeError, "UnicodeDecodeError"):
            prompt_loader.load_prompt("sample.md")

    def test_unexpected_missing_and_malformed_variables_fail(self) -> None:
        for text, expected in (("${unknown}", ()), ("没有变量", ("context",)), ("${broken", ())):
            self.write(text)
            with self.assertRaisesRegex(ValueError, "提示词变量"):
                prompt_loader.load_prompt("sample.md", expected)

    def test_paths_cannot_escape_prompt_root(self) -> None:
        for path in ("../outside.md", str(self.root.parent / "outside.md"), "sample.txt"):
            with self.assertRaisesRegex(ValueError, "路径必须"):
                prompt_loader.load_prompt(path)

    def test_cache_requires_explicit_reload_or_process_restart(self) -> None:
        self.write("原提示词")
        self.assertEqual(prompt_loader.load_prompt("sample.md"), "原提示词")
        self.write("新提示词")
        self.assertEqual(prompt_loader.load_prompt("sample.md"), "原提示词")
        prompt_loader.load_prompt.cache_clear()
        self.assertEqual(prompt_loader.load_prompt("sample.md"), "新提示词")

    def test_legacy_newlines_do_not_change_inserted_data(self) -> None:
        self.write("摘要：\n${transcript}")
        self.assertEqual(prompt_loader.render_prompt("sample.md", {"transcript": "甲\n乙"},
                                                    literal_newlines=True), "摘要：\\n甲\n乙")


class PromptMigrationTest(unittest.TestCase):
    def test_model_inputs_match_reviewed_snapshots(self) -> None:
        # 此快照专测旧字符上限文案；生产Token策略另行验证。
        item = patch.object(config, "memory_summary_max_tokens", None)
        item.start()
        self.addCleanup(item.stop)
        target = SimpleNamespace(target_id="target-test", document_format=SimpleNamespace(value="pdf"),
                                 locator="page:3", anomalies=[AnomalyType.MISSING_TEXT], evidence={"content": "check"})
        # 回答与解析保持节点7快照；节点8明确更新摘要规则和实际换行。
        snapshots = [
            (prompts.SYSTEM_PROMPT, "6acd53ea322d6b0f019ba4326ff8031368d3eb6b2b6d7348512bda091da1b7a4"),
            (prompts.build_answer_prompt().invoke({"history": [], "context": "context", "input": "question"}).to_string(),
             "fb0707f0771d58ee89ee92422e583b553c90f8f3c7399324b1687a3153d17676"),
            (build_review_prompt(target), "e99e6f4bbbcd43f5e620dccec77b5c03aac9b2061a70b3ab4d505aaaa9c39858"),
            (_build_summary_prompt("old", "dialogue"), "02d6e81db16ada377f2cadc6c949018bb0b50ce995384f4c51ee9a30989bbfa8"),
            (_build_summary_prompt("", ""), "95c66c5e0a260c76366ac1c4fa046811a9853d6a02e2bad85dcd2afa3f354916"),
        ]
        for text, expected in snapshots:
            self.assertEqual(hashlib.sha256(text.encode("utf-8")).hexdigest(), expected)

    def test_answer_roles_and_json_literals_remain_safe(self) -> None:
        with patch.object(prompts, "SYSTEM_PROMPT", '规则示例：{"key": 1}'), patch.object(
            prompts, "USER_PROMPT", '字面示例：{"key": 1}；价格$$10\n${context}\n${input}'
        ):
            messages = prompts.build_answer_prompt().invoke({
                "history": [], "context": '${input} {secret}', "input": "问题",
            }).to_messages()
        self.assertEqual([message.type for message in messages], ["system", "human"])
        self.assertEqual(messages[0].content, '规则示例：{"key": 1}')
        self.assertIn('字面示例：{"key": 1}；价格$10', messages[1].content)
        self.assertIn('${input} {secret}', messages[1].content)

    def test_both_system_and_user_changes_modify_fingerprint(self) -> None:
        original = prompts.answer_prompt_sha256()
        with patch.object(prompts, "SYSTEM_PROMPT", prompts.SYSTEM_PROMPT + "补充"):
            self.assertNotEqual(prompts.answer_prompt_sha256(), original)
        with patch.object(prompts, "USER_PROMPT", prompts.USER_PROMPT + "补充"):
            self.assertNotEqual(prompts.answer_prompt_sha256(), original)


if __name__ == "__main__":
    unittest.main()
