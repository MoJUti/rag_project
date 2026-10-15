"""表格展开保持单元格与来源，不对损坏表格猜列。"""

import json
import unittest

from generation.evidence import AnswerEvidence
from generation.table_context import expand_tables, named_cell_evidence_json


class TableContextTest(unittest.TestCase):
    def test_row_and_column_values_empty_cells_and_signs_are_preserved(self) -> None:
        text = "单位：%\n|月份|库存|进口|\n|---|---|---|\n|12月|47.8|-3|\n|4月||47.3|\n"
        expanded = expand_tables(text)
        rows = [json.loads(line) for line in expanded.splitlines() if line.startswith("{")]
        self.assertEqual(rows[0]["单元格"][1], {"列序": 2, "表头": "库存", "值": "47.8"})
        self.assertEqual(rows[0]["单元格"][2]["值"], "-3")
        self.assertEqual(rows[1]["单元格"][1]["值"], "")
        self.assertTrue(expanded.startswith("单位：%\n"))

    def test_duplicate_and_empty_headers_do_not_overwrite_columns(self) -> None:
        expanded = expand_tables("||值|值|\n|---|---|---|\n|对象|1|2|\n")
        row = json.loads(expanded.splitlines()[1])
        self.assertEqual([cell["表头"] for cell in row["单元格"]], ["", "值", "值"])
        self.assertEqual([cell["值"] for cell in row["单元格"]], ["对象", "1", "2"])

    def test_nonrectangular_or_escaped_table_remains_unchanged(self) -> None:
        for text in ("|甲|乙|\n|---|---|\n|值|\n", "|甲|乙|\n|---|---|\n|值\\|值|2|\n", "非表格\n原文"):
            self.assertEqual(expand_tables(text), text)

    def test_original_evidence_and_ids_are_not_mutated(self) -> None:
        text = "|月|值|\n|---|---|\n|12月|47.8|\n"
        evidence = AnswerEvidence("S1", "chunk1", "doc1", "原文.md", [{"page_number": 2}], text)
        rendered = json.loads(named_cell_evidence_json([evidence]))[0]
        self.assertEqual(evidence.text, text)
        self.assertEqual(rendered["citation_id"], "S1")
        self.assertEqual(rendered["locations"], evidence.locations)
        self.assertNotEqual(rendered["text"], text)


if __name__ == "__main__":
    unittest.main()
