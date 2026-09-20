"""模型配置与可选依赖提示测试。"""

import unittest
from unittest.mock import patch

from anitopy_ml.errors import ConfigurationError, SchemaValidationError
from anitopy_ml.modeling.config import ExtractorModelConfig
from anitopy_ml.modeling.runtime import require_training_dependencies
from anitopy_ml.schemas import BIO_LABELS


class ModelConfigTests(unittest.TestCase):
    """验证标签顺序固定且依赖缺失可读。"""

    def test_default_config_uses_project_bio_labels(self) -> None:
        config = ExtractorModelConfig()
        self.assertEqual(tuple(config.label_to_id()), BIO_LABELS)

    def test_changed_label_order_is_rejected(self) -> None:
        with self.assertRaises(SchemaValidationError):
            ExtractorModelConfig(labels=tuple(reversed(BIO_LABELS))).validate()

    def test_missing_dependency_has_chinese_instruction(self) -> None:
        with patch("anitopy_ml.modeling.runtime.importlib.import_module", side_effect=ModuleNotFoundError):
            with self.assertRaisesRegex(ConfigurationError, "uv sync --extra train"):
                require_training_dependencies()
