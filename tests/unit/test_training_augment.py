"""训练集专用增强测试。"""

import unittest

from anitopy_ml.errors import SchemaValidationError
from anitopy_ml.training.augment import augment_train_records, augment_train_whitespace


class TrainingAugmentTests(unittest.TestCase):
    """验证增强仅发生在训练组且保留追溯信息。"""

    def test_whitespace_augmentation_rebuilds_labels_and_lineage(self) -> None:
        record = {
            "sample_id": "样本1", "split": "train", "work_group_id": "作品组1",
            "text": "作品 EP01", "characters": list("作品 EP01"),
            "labels": ["O", "O", "O", "B-EPISODE_EXPR", "I-EPISODE_EXPR", "I-EPISODE_EXPR", "I-EPISODE_EXPR"],
        }
        augmented = augment_train_whitespace(record)
        self.assertEqual(augmented["parent_sample_id"], "样本1")
        self.assertEqual(augmented["work_group_id"], "作品组1")
        self.assertEqual(augmented["labels"][3], "O")
        self.assertEqual(augmented["text"], "作品  EP01")

    def test_validation_record_is_rejected_and_batch_skips_it(self) -> None:
        record = {"sample_id": "样本2", "split": "validation", "text": "作品 EP01", "characters": list("作品 EP01"), "labels": ["O"] * 7}
        with self.assertRaises(SchemaValidationError):
            augment_train_whitespace(record)
        self.assertEqual(augment_train_records([record]), [])
