"""训练控制与检查点元数据测试。"""

import unittest

from anitopy_ml.errors import SchemaValidationError
from anitopy_ml.training.control import EarlyStopping, TrainingProgress


class TrainingControlTests(unittest.TestCase):
    """验证早停策略与恢复数据版本校验。"""

    def test_early_stopping_resets_on_improvement(self) -> None:
        control = EarlyStopping(patience=2, min_delta=0.01)
        self.assertFalse(control.update(0.5))
        self.assertFalse(control.update(0.505))
        self.assertFalse(control.update(0.52))
        self.assertFalse(control.update(0.52))
        self.assertTrue(control.update(0.52))

    def test_progress_requires_sha256(self) -> None:
        progress = TrainingProgress(0, 0, None, {}, "a" * 64)
        self.assertEqual(progress.model_dump()["epoch"], 0)
        with self.assertRaises(SchemaValidationError):
            TrainingProgress(0, 0, None, {}, "short").validate()
