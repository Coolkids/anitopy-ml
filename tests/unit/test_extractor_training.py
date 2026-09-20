"""XLM-R 训练配置读取测试。"""

import unittest
from unittest.mock import patch

from anitopy_ml.errors import ConfigurationError
from anitopy_ml.training.extractor import read_extractor_training_config


class ExtractorTrainingTests(unittest.TestCase):
    """确保训练入口在加载大模型前先验证可复现配置。"""

    def test_reads_project_training_config(self) -> None:
        config = read_extractor_training_config("configs/抽取训练.json")
        self.assertEqual(config.learning_rate, 2e-5)
        self.assertEqual(config.early_stopping_patience, 5)

    def test_rejects_incomplete_config(self) -> None:
        with patch("anitopy_ml.training.extractor.Path.read_text", return_value='{"学习率": 0.1}'):
            with self.assertRaises(ConfigurationError):
                read_extractor_training_config("错误配置.json")
