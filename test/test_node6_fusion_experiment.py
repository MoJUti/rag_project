"""节点6词元和融合的核心边界测试。"""

import unittest

from scripts.run_node6_fusion_experiment import fuse_ids, mixed_tokenize


class FusionExperimentTest(unittest.TestCase):
    def test_mixed_tokenize_keeps_numbers_and_chinese_bigrams(self) -> None:
        self.assertEqual(
            mixed_tokenize("Qwen3.8 模型2025年"),
            ["qwen3.8", "模", "型", "模型", "2025", "年"],
        )

    def test_fuse_ids_deduplicates_and_can_weight_lexical(self) -> None:
        dense = ["a", "b", "c"]
        lexical = ["c", "b", "a"]
        self.assertEqual(fuse_ids(dense, lexical, 0.0, 3), dense)
        self.assertEqual(fuse_ids(dense, lexical, 2.0, 3)[0], "c")
        self.assertEqual(len(fuse_ids(dense, lexical, 1.0, 3)), 3)


if __name__ == "__main__":
    unittest.main()
