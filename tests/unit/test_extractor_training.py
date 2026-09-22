"""XLM-R 训练配置读取测试。"""

import unittest
from unittest.mock import patch

import torch

from anitopy_ml.errors import ConfigurationError, InputValidationError
from anitopy_ml.modeling.config import ExtractorModelConfig
from anitopy_ml.schemas import BIO_LABELS
from anitopy_ml.training.extractor import load_initial_model_weights, read_extractor_training_config


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

    def test_initial_weights_only_loads_compatible_model(self) -> None:
        original = torch.nn.Linear(2, 2)
        restored = torch.nn.Linear(2, 2)
        metadata = {
            "训练进度": {"training_manifest_sha256": "a" * 64},
            "模型配置": {"模型名称": "FacebookAI/xlm-roberta-base", "模型版本": "main", "标签": list(BIO_LABELS)},
        }
        with (
            patch("anitopy_ml.training.extractor.read_checkpoint_metadata", return_value=metadata),
            patch("anitopy_ml.training.extractor.require_training_dependencies", return_value=(torch, None)),
            patch("anitopy_ml.training.extractor.Path.is_file", return_value=True),
            patch("anitopy_ml.training.extractor.Path.read_bytes", return_value=b"checkpoint"),
            patch.object(torch, "load", return_value=original.state_dict()),
        ):
            provenance = load_initial_model_weights(
                restored,
                source_directory="来源目录",
                output_directory="新训练目录",
                model_config=ExtractorModelConfig(),
            )
        self.assertEqual(provenance["来源目录"], "来源目录")
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(original.parameters(), restored.parameters())))

    def test_initial_weights_rejects_overwriting_source_directory(self) -> None:
        with self.assertRaises(InputValidationError):
            load_initial_model_weights(
                torch.nn.Linear(2, 2),
                source_directory="同一目录",
                output_directory="同一目录",
                model_config=ExtractorModelConfig(),
            )
