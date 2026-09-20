"""作品分组与稳定数据划分测试。"""

import unittest

from anitopy_ml.data.grouping import audit_assignments, build_group_assignments, find_review_candidates
from anitopy_ml.data.ingest import TitleRecord


def record(sample_id: str, title: str) -> TitleRecord:
    """构造最小的可追溯标题记录。"""
    return TitleRecord(sample_id, 2, title, "a" * 64, sample_id.ljust(64, "0"))


class GroupingTests(unittest.TestCase):
    """验证作品组不跨分区且人工覆盖优先。"""

    def test_episode_variants_share_group_and_split(self) -> None:
        records = [
            record("样本1", "[组] 示例作品 S01E01 [1080p x264 AAC]"),
            record("样本2", "[组] 示例作品 S01E02 [1080p x264 AAC]"),
        ]
        assignments, report = build_group_assignments(records, seed=7)
        self.assertEqual(assignments[0].work_group_id, assignments[1].work_group_id)
        self.assertEqual(assignments[0].split, assignments[1].split)
        self.assertEqual(report["自动合并组数"], 1)

    def test_manual_override_has_priority_and_is_stable(self) -> None:
        records = [record("样本1", "甲作品 EP01"), record("样本2", "乙作品 EP01")]
        first, _ = build_group_assignments(records, seed=9, overrides={"样本2": "人工组"})
        second, _ = build_group_assignments(records, seed=9, overrides={"样本2": "人工组"})
        self.assertEqual(first, second)
        self.assertEqual(first[1].work_group_id, "人工组")
        self.assertEqual(first[1].reason, "manual_override")

    def test_cross_split_group_is_rejected(self) -> None:
        assignments, _ = build_group_assignments([record("样本1", "作品 EP01")])
        broken = [assignments[0], assignments[0].__class__("样本2", assignments[0].work_group_id, "test", "manual_override", "作品")]
        with self.assertRaises(ValueError):
            audit_assignments(broken)

    def test_leading_release_group_only_creates_review_candidate(self) -> None:
        records = [record("样本1", "[甲组] 示例作品 EP01"), record("样本2", "[乙组] 示例作品 EP02")]
        assignments, _ = build_group_assignments(records)
        candidates = find_review_candidates(records, assignments, similarity_threshold=0.85)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(candidates[0]["作品组ID"]), 2)

    def test_similar_but_nonidentical_keys_enter_review_queue(self) -> None:
        records = [record("样本1", "示例作品 第一季"), record("样本2", "示例作品 第二季")]
        assignments, _ = build_group_assignments(records)
        candidates = find_review_candidates(records, assignments, similarity_threshold=0.85)
        self.assertTrue(any("相似度" in item for item in candidates))
