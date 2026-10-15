"""复核与汇总必须保留错误案例、空分母和证据支持边界。"""

import copy
import hashlib
import unittest
from scripts.run_node7_prompt_ab import digest
from scripts.summarize_node7_prompt_ab import summarize, validate_review


class PromptABReviewTest(unittest.TestCase):
    def review(self, answer: str, claims: list[dict]) -> dict:
        return {"answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                "correctness": 2, "completeness": 2, "boundary": 2,
                "critical_errors": [], "claims": claims}

    def test_unknown_citation_can_be_scored_as_failure_but_not_support(self) -> None:
        row = {"raw_answer": "17台[S9]"}
        claims = [{"text": "17台", "cited_ids": ["S9"], "support": "unsupported", "requires_citation": True}]
        review = self.review(row["raw_answer"], claims)
        validate_review(row, review, {"S1"})
        claims[0]["support"] = "supported"
        with self.assertRaisesRegex(ValueError, "未知证据"):
            validate_review(row, review, {"S1"})

    def test_review_hash_excerpt_and_score_must_match_answer(self) -> None:
        row = {"raw_answer": "17台[S1]"}
        review = self.review(row["raw_answer"], [{"text": "17台", "cited_ids": ["S1"],
                                                 "support": "supported", "requires_citation": True}])
        for field, value in (("answer_sha256", "错误哈希"), ("correctness", True)):
            changed = {**review, field: value}
            with self.assertRaises(ValueError):
                validate_review(row, changed, {"S1"})
        changed = copy.deepcopy(review)
        changed["claims"][0]["text"] = "999台"
        with self.assertRaisesRegex(ValueError, "原始答案"):
            validate_review(row, changed, {"S1"})

    def test_no_required_claims_is_not_one_hundred_percent(self) -> None:
        fixture = {"question_id": "q", "group": "test", "evidence": []}
        report = {"identity": {"fixtures": [fixture]}, "rows": {}}
        reviews = {"experiment_sha256": digest(report["identity"]), "reviewer": "测试", "rows": {}}
        for version in ("baseline", "candidate"):
            key = f"{version}/q"
            report["rows"][key] = {"status": "success", "raw_answer": "资料不足", "model_info": {}, "latency_ms": 1}
            reviews["rows"][key] = self.review("资料不足", [])
        result = summarize(report, reviews)
        group = result["versions"]["baseline"]["groups"]["test"]
        self.assertIsNone(group["citation_precision"])
        self.assertIsNone(group["citation_coverage"])
        report["rows"].pop("candidate/q")
        reviews["rows"].pop("candidate/q")
        with self.assertRaisesRegex(ValueError, "尚未完整"):
            summarize(report, reviews)
