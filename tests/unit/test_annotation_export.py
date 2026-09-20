"""字符级BIO训练导出测试。"""

import unittest

from anitopy_ml.annotation.export import audit_effective_work_groups, build_training_splits, payload_to_bio
from anitopy_ml.annotation.review import review_sample
from anitopy_ml.annotation.seed import build_seed_record
from anitopy_ml.annotation.store import AnnotationStore
from anitopy_ml.errors import SchemaValidationError


class AnnotationExportTests(unittest.TestCase):
    """验证训练导出保留训练许可和BIO边界。"""

    def test_accepted_record_exports_bio(self) -> None:
        with AnnotationStore(":memory:") as store:
            store.save(build_seed_record("样本1", "作品 EP01"), actor="规则")
            review_sample(store, "样本1", actor="审核员", decision="approve")
            payload = store.latest_payload("样本1")
        item = payload_to_bio(payload, "train")
        self.assertEqual(item["labels"][3], "B-EPISODE_EXPR")

    def test_unaccepted_record_is_excluded(self) -> None:
        record = build_seed_record("样本2", "作品 EP01").model_dump()
        splits, skipped = build_training_splits([record], {"样本2": "train"})
        self.assertEqual(splits["train"], [])
        self.assertEqual(sum(skipped.values()), 1)
        with self.assertRaises(SchemaValidationError):
            payload_to_bio(record, "train")

    def test_manual_group_crossing_frozen_splits_is_rejected(self) -> None:
        with self.assertRaises(SchemaValidationError):
            audit_effective_work_groups(
                {"样本1": "train", "样本2": "test"},
                {"样本1": "人工组-同一作品", "样本2": "人工组-同一作品"},
            )
