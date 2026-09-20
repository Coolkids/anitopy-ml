"""训练器配置测试。"""

import unittest

from anitopy_ml.errors import SchemaValidationError
from anitopy_ml.training.trainer import TrainerConfig


class TrainerConfigTests(unittest.TestCase):
    """验证不依赖 Torch 的训练配置校验。"""

    def test_default_config_is_valid(self) -> None:
        TrainerConfig().validate()

    def test_invalid_accumulation_is_rejected(self) -> None:
        with self.assertRaises(SchemaValidationError):
            TrainerConfig(gradient_accumulation_steps=0).validate()
