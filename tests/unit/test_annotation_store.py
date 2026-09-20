"""SQLite标注存储和规则预标注测试。"""

import unittest

from anitopy_ml.annotation.seed import build_seed_record
from anitopy_ml.annotation.review import mark_unresolved, review_sample, save_manual_revision
from anitopy_ml.annotation.store import AnnotationStore
from anitopy_ml.errors import SchemaValidationError
from anitopy_ml.schemas import AnnotationMetadata


class AnnotationStoreTests(unittest.TestCase):
    """验证标注版本、审计和弱标注状态。"""

    def test_seed_record_is_pending_and_not_trainable(self) -> None:
        record = build_seed_record("样本-1", "示例作品 S01E03 1080p")
        self.assertEqual(record.annotation.tier, "weak")
        self.assertEqual(record.annotation.review_status, "pending")
        self.assertFalse(record.lineage.training_allowed)
        self.assertTrue(record.spans)

    def test_seed_handles_season_episode_without_overlap(self) -> None:
        record = build_seed_record("样本-季集", "示例作品 S01E03")
        record.validate()
        self.assertEqual([(item.label, item.text) for item in record.spans], [("SEASON_EXPR", "S01"), ("EPISODE_EXPR", "E03")])

    def test_store_rejects_stale_version(self) -> None:
        record = build_seed_record("样本-2", "示例作品 EP01")
        with AnnotationStore(":memory:") as store:
            store.save(record, actor="测试人员")
            self.assertEqual(store.pending_sample_ids(), ["样本-2"])
            self.assertEqual(store.latest_payload("样本-2")["sample_id"], "样本-2")
            with self.assertRaises(SchemaValidationError):
                store.save(record, actor="测试人员")

    def test_next_version_can_be_saved(self) -> None:
        record = build_seed_record("样本-3", "示例作品 OVA")
        with AnnotationStore(":memory:") as store:
            store.save(record, actor="规则预标注")
            record.annotation = AnnotationMetadata("gold", "accepted", "审核员甲", version=2)
            store.save(record, actor="审核员甲", action="审核通过")
            self.assertEqual(store.latest_payload("样本-3")["annotation"]["version"], 2)

    def test_review_approval_promotes_record_and_keeps_audit_version(self) -> None:
        record = build_seed_record("样本-4", "示例作品 EP01")
        with AnnotationStore(":memory:") as store:
            store.save(record, actor="规则预标注")
            approved = review_sample(store, "样本-4", actor="审核员甲", decision="approve")
            self.assertEqual(approved.annotation.tier, "gold")
            self.assertEqual(approved.annotation.version, 2)
            self.assertTrue(approved.lineage.training_allowed)
            self.assertIn("human", approved.lineage.sources)
            self.assertEqual(store.pending_sample_ids(), [])

    def test_unresolved_sample_stays_in_review_and_cannot_train(self) -> None:
        record = build_seed_record("样本-无法判断", "无法确认的标题")
        with AnnotationStore(":memory:") as store:
            store.save(record, actor="规则预标注")
            unresolved = mark_unresolved(store, "样本-无法判断", actor="审核员甲")
            queue = store.review_queue(statuses=("needs_review",))
        self.assertEqual(unresolved.annotation.review_status, "needs_review")
        self.assertFalse(unresolved.lineage.training_allowed)
        self.assertEqual(unresolved.annotation.unlabeled_regions, ((0, len("无法确认的标题")),))
        self.assertEqual(queue[0]["sample_id"], "样本-无法判断")

    def test_manual_work_group_override_is_saved(self) -> None:
        record = build_seed_record("样本-作品组", "示例作品 EP01")
        with AnnotationStore(":memory:") as store:
            store.save(record, actor="规则预标注")
            store.set_work_group("样本-作品组", "作品组-示例", actor="审核员甲")
            self.assertEqual(store.work_group_id("样本-作品组"), "作品组-示例")

    def test_manual_revision_keeps_raw_text_and_requires_new_version(self) -> None:
        record = build_seed_record("样本-5", "示例作品 EP01")
        with AnnotationStore(":memory:") as store:
            store.save(record, actor="规则预标注")
            payload = store.latest_payload("样本-5")
            payload["spans"] = []
            revised = save_manual_revision(store, "样本-5", payload, actor="审核员甲")
            self.assertEqual(revised.annotation.version, 2)
            self.assertEqual(revised.annotation.review_status, "needs_review")
            self.assertFalse(revised.lineage.training_allowed)
