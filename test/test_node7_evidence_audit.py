"""证据诊断不可跨来源、拼错行、吞掉数值差异或冒充全量准确率。"""

import copy
import hashlib
import unittest

from scripts.audit_node7_evidence import add_reviews, audit, contains_quote, match_requirement
from scripts.run_node7_prompt_ab import digest


class EvidenceAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.requirement = {"id": "出勤人数", "document_ids": ["item"], "alternatives": [
            {"quote": "出勤 | 800", "context": ["今日考勤"]},
        ]}
        self.document = {"chunk_id": "c1", "document_id": "doc1", "citation_id": "S1",
                         "source": "测试表.xlsx", "text": "今日考勤\n出勤 | 800", "locations": []}
        self.catalog = {"c1": {"document_id": "doc1", "text": self.document["text"]}}
        self.report = {"identity": {"catalog_sha256": digest(self.catalog)}, "rows": {"q": {
            "status": "success", "raw_answer": "出勤800人[S1]", "evidence": [self.document],
            "question": {"relevant_document_ids": ["item"]}, "retrieval_metrics": {"evidence_recall_at_6": 0},
        }}}
        self.specification = {"experiment_sha256": digest(self.report["identity"]),
                              "report_content_sha256": digest(self.report), "reviewer": "测试复核",
                              "evidence_cases": {"q": {"scope_note": "诊断一项事实，不是整体答案评分",
                                                       "requirements": [self.requirement]}}, "semantic_reviews": {}}

    def test_preserves_numeric_boundaries_sign_and_units(self) -> None:
        self.assertFalse(contains_quote("出勤 | 8000", "出勤 | 800"))
        self.assertFalse(contains_quote("速动比率0.400", "速动比率0.40"))
        self.assertFalse(contains_quote("金额80元", "金额80万元"))
        self.assertFalse(contains_quote("现金流-65", "现金流65"))
        self.assertFalse(contains_quote("金额8000", "800"))
        self.assertTrue(contains_quote("**出勤** | ８００", "出勤 | 800"))
        self.assertTrue(contains_quote("营业收入462,903,179.20 461,194,576.50", "营业收入462,903,179.20"))
        self.assertTrue(contains_quote("16:00-17:00\nparty地点", "16:00-17:00"))
        with self.assertRaisesRegex(ValueError, "不能为空"):
            contains_quote("任何文本", " ** ")

    def test_same_number_in_wrong_source_or_row_does_not_match(self) -> None:
        wrong_source = {**self.document, "document_id": "doc2"}
        wrong_row = {**self.document, "text": "今日考勤\n出勤 | 700\n外勤 | 800"}
        self.assertEqual(match_requirement(self.requirement, [wrong_source, wrong_row], {"item": "doc1"}), [])
        self.assertEqual(len(match_requirement(self.requirement, [self.document], {"item": "doc1"})), 1)

    def test_context_and_fact_cannot_be_stitched_across_chunks(self) -> None:
        documents = [{**self.document, "text": "今日考勤"}, {**self.document, "chunk_id": "c2", "text": "出勤 | 800"}]
        self.assertEqual(match_requirement(self.requirement, documents, {"item": "doc1"}), [])

    def test_partial_diagnosis_preserves_old_scores_and_no_overall_percent(self) -> None:
        frozen = copy.deepcopy(self.report)
        result = audit(self.report, self.specification, self.catalog, {"item": "doc1"})
        self.assertEqual(result["evidence_rows"]["q"]["diagnosis"], "minimal_spans_present")
        self.assertEqual(result["evidence_rows"]["q"]["old_metrics"]["evidence_recall_at_6"], 0)
        self.assertEqual(result["summary"]["semantic_reviewed"], 0)
        self.assertEqual(result["summary"]["semantic_unreviewed"], 1)
        self.assertIsNone(result["summary"]["overall_answer_accuracy"])
        self.assertIsNone(result["summary"]["overall_corrected_recall"])
        self.assertEqual(self.report, frozen)

    def test_stale_report_catalog_and_context_fail_explicitly(self) -> None:
        changed = copy.deepcopy(self.report)
        changed["rows"]["q"]["raw_answer"] = "被改写的答案"
        with self.assertRaisesRegex(ValueError, "报告内容指纹"):
            audit(changed, self.specification, self.catalog, {"item": "doc1"})
        with self.assertRaisesRegex(ValueError, "切块指纹"):
            audit(self.report, self.specification, {}, {"item": "doc1"})
        changed = copy.deepcopy(self.report)
        changed["rows"]["q"]["evidence"][0]["text"] = "伪造证据"
        spec = {**self.specification, "report_content_sha256": digest(changed)}
        with self.assertRaisesRegex(ValueError, "上下文与冻结切块不符"):
            audit(changed, spec, self.catalog, {"item": "doc1"})

    def test_duplicate_requirements_unknown_and_out_of_scope_sources_fail(self) -> None:
        for case in (
            {"requirements": [self.requirement, self.requirement], "scope_note": "重复"},
            {"requirements": [{**self.requirement, "document_ids": ["wrong"]}], "scope_note": "越界"},
        ):
            with self.assertRaises(ValueError):
                audit(self.report, {**self.specification, "evidence_cases": {"q": case}}, self.catalog, {"item": "doc1"})
        with self.assertRaisesRegex(ValueError, "来源不在清单"):
            match_requirement(self.requirement, [self.document], {})

    def test_semantic_review_requires_exact_answer_and_real_source_quote(self) -> None:
        review = {"answer_sha256": hashlib.sha256(self.report["rows"]["q"]["raw_answer"].encode()).hexdigest(),
                  "finding": "supported_abstention", "answer_excerpt": "出勤800人", "notes": "仅用于测试字段校验",
                  "source_checks": [{"citation_id": "S1", "quote": "出勤 | 800"}]}
        spec = {**self.specification, "semantic_reviews": {"q": review}}
        self.assertEqual(audit(self.report, spec, self.catalog, {"item": "doc1"})["summary"]["semantic_reviewed"], 1)
        for field, value in (("answer_sha256", "旧答案"), ("answer_excerpt", "900人"),
                             ("source_checks", [{"citation_id": "S9", "quote": "出勤 | 800"}]),
                             ("source_checks", [{"citation_id": "S1", "quote": "出勤 | 900"}]),
                             ("finding", "typo")):
            with self.assertRaises(ValueError):
                audit(self.report, {**spec, "semantic_reviews": {"q": {**review, field: value}}},
                      self.catalog, {"item": "doc1"})

    def additional(self) -> dict:
        return {"experiment_sha256": digest(self.report["identity"]), "report_content_sha256": digest(self.report),
                "reviewer": "测试逐题复核", "review_scope": "核心事实，不是穷尽引用评分",
                "semantic_reviews": {"q": {"finding": "supported_answer", "answer_excerpt": "出勤800人",
                    "notes": "人工核对测试事实", "source_checks": [{"citation_id": "S1", "quote": "出勤 | 800"}]}}}

    def test_additional_reviews_bind_hash_and_do_not_mutate_previous_specification(self) -> None:
        before = copy.deepcopy(self.specification)
        combined = add_reviews(self.report, self.specification, self.additional())
        self.assertEqual(self.specification, before)
        self.assertEqual(combined["semantic_reviews"]["q"]["answer_sha256"],
                         hashlib.sha256(self.report["rows"]["q"]["raw_answer"].encode()).hexdigest())
        result = audit(self.report, combined, self.catalog, {"item": "doc1"})
        self.assertEqual(result["summary"]["semantic_unreviewed"], 0)
        self.assertEqual(result["summary"]["findings"], {"supported_answer": 1})
        self.assertIn("不是穷尽", result["review_scope"])
        self.assertIsNone(result["summary"]["overall_answer_accuracy"])

    def test_additional_reviews_reject_partial_unknown_duplicate_and_forged_hash(self) -> None:
        extra = self.additional()
        variants = [
            {**extra, "semantic_reviews": {}},
            {**extra, "semantic_reviews": {"unknown": extra["semantic_reviews"]["q"]}},
            {**extra, "report_content_sha256": "旧报告"},
            {**extra, "semantic_reviews": {"q": {**extra["semantic_reviews"]["q"], "answer_sha256": "预填哈希"}}},
        ]
        for changed in variants:
            with self.assertRaises(ValueError):
                add_reviews(self.report, self.specification, changed)
        base = {**self.specification, "semantic_reviews": {"q": extra["semantic_reviews"]["q"]}}
        with self.assertRaisesRegex(ValueError, "题号重复"):
            add_reviews(self.report, base, extra)

    def test_additional_review_does_not_relax_original_source_validation(self) -> None:
        extra = self.additional()
        extra["semantic_reviews"]["q"]["source_checks"][0]["quote"] = "出勤 | 900"
        combined = add_reviews(self.report, self.specification, extra)
        with self.assertRaisesRegex(ValueError, "原文不存在"):
            audit(self.report, combined, self.catalog, {"item": "doc1"})
