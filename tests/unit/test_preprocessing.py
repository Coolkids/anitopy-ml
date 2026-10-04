"""验证按顺序执行的名称预处理规则。"""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from anitopy_ml.errors import ConfigurationError, PreprocessingError
from anitopy_ml.processing.config import parse_processing_config
from anitopy_ml.processing.preprocessing import PreprocessingPipeline


class PreprocessingTests(unittest.TestCase):
    """用独立输入验证规则顺序、匹配语义、限额和错误处理。"""

    def setUp(self) -> None:
        example_path = Path(__file__).resolve().parents[2] / "configs" / "processing.example.toml"
        self.payload = tomllib.loads(example_path.read_text(encoding="utf-8"))

    def test_four_rule_example_runs_in_order(self) -> None:
        self.payload["preprocessing"]["enabled"] = True
        result = PreprocessingPipeline(parse_processing_config(self.payload)).apply(
            "[广告]示例作品_第03集[转载:示例站].mkv",
            trace=True,
        )
        self.assertEqual(result.processed_text, "示例作品 E03.mkv")
        self.assertTrue(result.changed)
        self.assertEqual([step.matched for step in result.steps], [1, 1, 1, 1])
        self.assertEqual(result.raw_text, "[广告]示例作品_第03集[转载:示例站].mkv")

    def test_count_limits_single_pass_and_next_rule_sees_output(self) -> None:
        config = parse_processing_config(
            {
                "config_version": 1,
                "preprocessing": {
                    "enabled": True,
                    "rules": [
                        {"id": "one", "type": "string_replace", "match": "A", "replacement": "AA", "count": 1},
                        {"id": "next", "type": "string_replace", "match": "AA", "replacement": "B", "count": 1},
                    ],
                },
            }
        )
        self.assertEqual(PreprocessingPipeline(config).apply("AAA").processed_text, "BAA")

    def test_string_match_is_literal_and_regex_flags_apply(self) -> None:
        config = parse_processing_config(
            {
                "config_version": 1,
                "preprocessing": {
                    "enabled": True,
                    "rules": [
                        {"id": "literal", "type": "string_replace", "match": ".", "replacement": "!"},
                        {"id": "regex", "type": "regex_replace", "pattern": "a", "replacement": "x", "flags": ["IGNORECASE"]},
                    ],
                },
            }
        )
        self.assertEqual(PreprocessingPipeline(config).apply("A.a").processed_text, "x!x")

    def test_zero_width_empty_output_and_expansion_fail_before_inference(self) -> None:
        config = parse_processing_config(
            {
                "config_version": 1,
                "preprocessing": {
                    "enabled": True,
                    "rules": [{"id": "lookahead", "type": "regex_mask", "pattern": "(?=A)"}],
                },
            }
        )
        with self.assertRaisesRegex(PreprocessingError, "零宽"):
            PreprocessingPipeline(config).apply("A")

        empty_config = parse_processing_config(
            {
                "config_version": 1,
                "preprocessing": {
                    "enabled": True,
                    "rules": [{"id": "erase", "type": "string_mask", "match": "A"}],
                },
            }
        )
        with self.assertRaises(PreprocessingError) as caught:
            PreprocessingPipeline(empty_config).apply("A")
        self.assertEqual(caught.exception.code, "PREPROCESSING_EMPTY")

        limited = parse_processing_config(
            {
                "config_version": 1,
                "preprocessing": {
                    "enabled": True,
                    "max_output_length": 2,
                    "rules": [{"id": "expand", "type": "string_replace", "match": "A", "replacement": "XYZ"}],
                },
            }
        )
        with self.assertRaises(PreprocessingError) as caught:
            PreprocessingPipeline(limited).apply("A")
        self.assertEqual(caught.exception.code, "PREPROCESSING_LIMIT")

    def test_rejects_zero_width_empty_string_patterns_during_load(self) -> None:
        with self.assertRaises(ConfigurationError):
            parse_processing_config(
                {
                    "config_version": 1,
                    "preprocessing": {
                        "enabled": False,
                        "rules": [{"id": "empty", "type": "regex_mask", "pattern": "a*"}],
                    },
                }
            )
