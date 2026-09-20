"""离线解析接口测试。"""

import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from anitopy_ml.api import MediaParser
from anitopy_ml.inference.character import CharacterTagger
from anitopy_ml.schemas import BIO_LABELS, ParseResult


class ApiTests(unittest.TestCase):
    """验证本地检查点加载和批量错误隔离。"""

    def _checkpoint_payload(self) -> dict[str, object]:
        model = CharacterTagger(3, len(BIO_LABELS))
        return {
            "模型": model.state_dict(),
            "词表": {"<填充>": 0, "<未知>": 1, "示": 2},
            "标签": list(BIO_LABELS),
        }

    def test_local_checkpoint_parses_and_batch_records_input_error(self) -> None:
        with (
            patch("anitopy_ml.api.require_training_dependencies", return_value=(torch, None)),
            patch("anitopy_ml.api.Path.is_file", return_value=True),
            patch.object(torch, "load", return_value=self._checkpoint_payload()),
        ):
            parser = MediaParser.from_pretrained(Path("模型.pt"))
            result = parser.parse("示例 S01E02")
            batch = parser.parse_batch(["示例", ""])
        self.assertIsInstance(result, ParseResult)
        self.assertEqual(result.raw_text, "示例 S01E02")
        self.assertIsInstance(batch[0], ParseResult)
        self.assertEqual(batch[1]["status"], "error")
