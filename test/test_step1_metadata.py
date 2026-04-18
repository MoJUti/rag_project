import unittest
from pathlib import Path

from core import config as c
from ingestion.legal_chunker import build_chunks_from_article_units, parse_legal_article_units
from ingestion.legal_preprocess import preprocess_legal_text


class TestStep1Metadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = Path("data/刑法.txt").read_text(encoding="utf-8")
        cls.cleaned = preprocess_legal_text(raw)
        cls.article_units = parse_legal_article_units(cls.cleaned)
        cls.article_to_chapter = {
            u["article_no"]: u["chapter"] for u in cls.article_units if u["article_no"]
        }
        cls.chunks, cls.metas = build_chunks_from_article_units(
            article_units=cls.article_units,
            max_chars=c.chunk_size,
            overlap_articles=0,
        )

    def test_print_metadata_preview(self):
        total = len(self.metas)
        miss_part = sum(1 for m in self.metas if not m["part"])
        miss_chapter = sum(1 for m in self.metas if not m["chapter"])
        miss_article = sum(1 for m in self.metas if not m["article_no"])

        print(
            "chunks={}, miss_part={} ({:.1%}), miss_chapter={} ({:.1%}), miss_article={} ({:.1%})".format(
                total,
                miss_part,
                miss_part / total,
                miss_chapter,
                miss_chapter / total,
                miss_article,
                miss_article / total,
            )
        )
        print("---first12---")
        for i, m in enumerate(self.metas[:12], 1):
            print(
                "[{}] part={} | chapter={} | section={} | article={} -> end={} | chunk_end={}".format(
                    i,
                    m["part"][:18],
                    m["chapter"][:24],
                    m["section"][:20],
                    m["article_no"],
                    m["article_end"],
                    m["chunk_article_end"],
                )
            )

        self.assertGreater(total, 0)

    def test_first_chapter_end_should_be_article_12(self):
        first = self.metas[0]
        self.assertEqual(first["chapter"], "第一章　刑法的任务、基本原则和适用范围")
        self.assertEqual(first["article_no"], "第一条")
        # 用户关注点：第一章应覆盖到第十二条
        self.assertEqual(first["article_end"], "第十二条")

    def test_first_chapter_chunks_should_cover_until_article_12(self):
        first_chapter_chunks = [
            m for m in self.metas if m["chapter"] == "第一章　刑法的任务、基本原则和适用范围"
        ]
        self.assertGreater(len(first_chapter_chunks), 0)
        chunk_ends = [m["chunk_article_end"] for m in first_chapter_chunks]
        self.assertIn("第十二条", chunk_ends)

    def test_chunk_should_not_cross_chapter(self):
        for m in self.metas:
            start_chapter = self.article_to_chapter.get(m["article_no"], "")
            end_chapter = self.article_to_chapter.get(m["chunk_article_end"], "")
            self.assertEqual(start_chapter, m["chapter"])
            self.assertEqual(end_chapter, m["chapter"])

    def test_required_fields_should_exist(self):
        required_fields = [
            "part",
            "chapter",
            "section",
            "article_no",
            "article_end",
            "chunk_article_end",
        ]
        sample = self.metas[0]
        for field in required_fields:
            self.assertIn(field, sample)

    def test_coverage_should_be_high(self):
        total = len(self.metas)
        miss_part = sum(1 for m in self.metas if not m["part"])
        miss_chapter = sum(1 for m in self.metas if not m["chapter"])
        miss_article = sum(1 for m in self.metas if not m["article_no"])

        self.assertLessEqual(miss_part, int(total * 0.05))
        self.assertLessEqual(miss_chapter, int(total * 0.05))
        self.assertLessEqual(miss_article, int(total * 0.05))


if __name__ == "__main__":
    unittest.main()
