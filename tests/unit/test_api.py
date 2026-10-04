"""离线解析接口测试。"""

import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from anitopy_ml.api import MediaParser
from anitopy_ml.inference.character import CharacterTagger
from anitopy_ml.processing.config import parse_processing_config
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

    def _processing_parser(self, rules: list[dict[str, object]]) -> MediaParser:
        config = parse_processing_config(
            {
                "config_version": 1,
                "preprocessing": {"enabled": True, "rules": rules},
            }
        )
        return MediaParser(
            model=object(),
            vocabulary=None,
            tokenizer=object(),
            labels=BIO_LABELS,
            device="cpu",
            model_version="测试模型",
            processing_config=config,
        )

    def test_parse_preprocesses_once_and_restores_original_text(self) -> None:
        parser = self._processing_parser(
            [{"id": "rename", "type": "string_replace", "match": "_", "replacement": " "}]
        )
        fake_result = ParseResult(raw_text="示例 标题")
        with (
            patch.object(parser, "_infer_title", return_value=fake_result) as infer,
            patch.object(parser._result_processor, "process", wraps=parser._result_processor.process) as process,
        ):
            result = parser.parse("示例_标题")

        infer.assert_called_once_with("示例 标题", apply_calibration=False)
        process.assert_called_once()
        self.assertEqual(result.raw_text, "示例_标题")
        self.assertEqual(result.preprocessing.processed_text, "示例 标题")
        self.assertEqual(result.schema_version, "1.1")

    def test_preprocessing_batch_preserves_errors_and_32_item_chunking(self) -> None:
        parser = self._processing_parser(
            [
                {"id": "rename", "type": "string_replace", "match": "_", "replacement": " "},
                {"id": "erase", "type": "string_mask", "match": "DROP"},
            ]
        )

        def infer_batch(titles: list[str], *, apply_calibration: bool = True) -> list[ParseResult]:
            self.assertFalse(apply_calibration)
            return [ParseResult(raw_text=title) for title in titles]

        titles = [f"title_{index}" for index in range(35)]
        titles.insert(7, "DROP")
        with (
            patch.object(parser, "_parse_extractor_batch", side_effect=infer_batch) as infer,
            patch.object(parser._result_processor, "process", wraps=parser._result_processor.process) as process,
        ):
            results = parser.parse_batch(titles, on_error="record")

        self.assertEqual([call.args[0].__len__() for call in infer.call_args_list], [32, 3])
        process.assert_called_once()
        self.assertEqual(len(results), len(titles))
        self.assertEqual(results[7]["status"], "error")
        self.assertEqual(results[7]["raw_text"], "DROP")
        for index, result in enumerate(results):
            if index == 7:
                continue
            self.assertEqual(result.raw_text, titles[index])
            self.assertEqual(result.preprocessing.processed_text, titles[index].replace("_", " "))
