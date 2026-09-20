"""词元偏移和长标题窗口测试。"""

import unittest

from anitopy_ml.training.alignment import IGNORE_LABEL_ID, align_bio_offsets
from anitopy_ml.training.windows import make_character_windows


class TrainingAlignmentTests(unittest.TestCase):
    """验证边界词元和重叠窗口不会产生错误监督。"""

    def test_offsets_assign_begin_inside_and_special_tokens(self) -> None:
        result = align_bio_offsets(
            "作品EP01",
            ["O", "O", "B-EPISODE_EXPR", "I-EPISODE_EXPR", "I-EPISODE_EXPR", "I-EPISODE_EXPR"],
            [(0, 0), (0, 2), (2, 4), (4, 6), (0, 0)],
            {"O": 0, "B-EPISODE_EXPR": 1, "I-EPISODE_EXPR": 2},
        )
        self.assertEqual(result.label_ids, (IGNORE_LABEL_ID, 0, 1, 2, IGNORE_LABEL_ID))
        self.assertFalse(result.issues)

    def test_cross_boundary_token_is_ignored(self) -> None:
        result = align_bio_offsets(
            "作品EP",
            ["O", "O", "B-EPISODE_EXPR", "I-EPISODE_EXPR"],
            [(1, 3)],
            {"O": 0, "B-EPISODE_EXPR": 1, "I-EPISODE_EXPR": 2},
        )
        self.assertEqual(result.label_ids, (IGNORE_LABEL_ID,))
        self.assertTrue(result.issues)

    def test_overlap_is_only_scored_once(self) -> None:
        windows = make_character_windows("abcdef", ["O"] * 6, max_chars=4, stride=2)
        self.assertEqual([(item.global_start, item.global_end) for item in windows], [(0, 4), (2, 6)])
        self.assertEqual(windows[1].loss_mask, (False, False, True, True))
