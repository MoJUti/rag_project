"""固定轨迹旧值诊断及摘要模板更新回归，不把正则识别当作模型语义测试。"""

import unittest

from memory.history_store import _build_summary_prompt
from scripts.regress_memory_forgetting import audit_forgotten_values


class MemoryForgettingRegressionTests(unittest.TestCase):
    def test_old_date_variants_and_revoked_mentions_are_detected(self) -> None:
        for text in ("原日期2027年2月19日已作废", "2027-02-19", "2027/2/19", "２０２７．２．１９"):
            self.assertTrue(audit_forgotten_values(text)["old_date_present"])

    def test_old_code_variants_are_detected(self) -> None:
        for text in ("ACL-7已撤销", "acl_7", "ACL—7", "ＡＣＬ－７"):
            self.assertTrue(audit_forgotten_values(text)["old_code_present"])
        self.assertFalse(audit_forgotten_values("另一个编码ACL-70仍有效")["old_code_present"])

    def test_clean_summary_keeps_valid_conditions(self) -> None:
        text = "青岚预算6319元，日期和审批码已撤销，目前未定；禁止外发。松屿预算18341元。"
        self.assertTrue(audit_forgotten_values(text)["passed"])

    def test_prompt_contains_cleanup_rule_without_removing_input_data(self) -> None:
        transcript = "原日期2027年2月19日，请忘记。"
        prompt = _build_summary_prompt("ACL-7已被要求忘记", transcript)
        self.assertIn("若旧摘要已说某条件被要求忘记", prompt)
        self.assertIn("只撤销用户指定对象、字段和范围", prompt)
        self.assertIn(transcript, prompt)
        self.assertIn("ACL-7已被要求忘记", prompt)
