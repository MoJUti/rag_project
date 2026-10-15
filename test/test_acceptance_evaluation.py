"""验收题的等价来源、列头和拒答评分边界。"""

import unittest

from scripts.build_node6_acceptance import build
from scripts.evaluate_node6_acceptance import score_question


class AcceptanceEvaluationTest(unittest.TestCase):
    def test_equivalent_copy_is_not_a_second_required_document(self) -> None:
        question = {"answerable": True, "relevant_document_groups": [["a", "copy"]],
                    "evidence": [{"document_ids": ["a", "copy"], "quote": "现金增加额 487",
                                  "context_quotes": ["2025E"], "locator": {"type": "line"}}]}
        catalog = {"c": {"document_id": "doc-copy", "text": "2025E\n现金增加额 487"}}
        result = score_question(question, ["c"], catalog, {"a": "doc-a", "copy": "doc-copy"})
        self.assertEqual(result["document_recall_at_6"], 1.0)
        self.assertEqual(result["complete_evidence_at_6"], 1.0)

    def test_matching_value_without_required_header_is_incomplete(self) -> None:
        question = {"answerable": True, "relevant_document_groups": [["a"]],
                    "evidence": [{"document_ids": ["a"], "quote": "经营现金流 102",
                                  "context_quotes": ["2024E"], "locator": {"type": "line"}}]}
        catalog = {"c": {"document_id": "doc-a", "text": "经营现金流 102"}}
        result = score_question(question, ["c"], catalog, {"a": "doc-a"})
        self.assertEqual(result["document_recall_at_6"], 1.0)
        self.assertEqual(result["evidence_recall_at_6"], 0.0)

    def test_refusal_is_not_scored_as_zero_recall(self) -> None:
        with self.assertRaisesRegex(ValueError, "不适用召回率"):
            score_question({"answerable": False}, [], {}, {})

    def test_curated_labels_resolve_to_original_sources(self) -> None:
        rows = build()
        self.assertEqual(len(rows), 23)
        self.assertEqual(sum(row["answerable"] for row in rows), 19)
        self.assertEqual(len({row["question_id"] for row in rows}), 23)
        for row in rows:
            if row["answerable"]:
                self.assertTrue(row["evidence"])
                self.assertTrue(all(atom["quote"] and atom["source_sha256"] for atom in row["evidence"]))


if __name__ == "__main__":
    unittest.main()
