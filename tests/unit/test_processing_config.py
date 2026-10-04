"""验证名称处理配置加载、字段契约与稳定指纹。"""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from anitopy_ml.errors import ConfigurationError
from anitopy_ml.processing.config import (
    load_processing_config,
    parse_processing_config,
)


class ProcessingConfigTests(unittest.TestCase):
    """覆盖示例文件和有业务意义的配置错误。"""

    def setUp(self) -> None:
        self.example_path = Path(__file__).resolve().parents[2] / "configs" / "processing.example.toml"
        self.payload = tomllib.loads(self.example_path.read_text(encoding="utf-8"))

    def test_example_describes_four_ordered_rules_and_default_off(self) -> None:
        config = load_processing_config(self.example_path)
        self.assertFalse(config.enabled)
        self.assertEqual(
            [rule.type for rule in config.rules],
            ["string_replace", "regex_replace", "string_mask", "regex_mask"],
        )
        self.assertEqual(len(config.fingerprint), 64)
        self.assertIsNotNone(config.source_path)

    def test_fingerprint_tracks_rule_order_but_not_source_path(self) -> None:
        first = parse_processing_config(self.payload, source_path="first.toml")
        second = parse_processing_config(self.payload, source_path="second.toml")
        self.assertEqual(first.fingerprint, second.fingerprint)

        reversed_payload = tomllib.loads(self.example_path.read_text(encoding="utf-8"))
        reversed_payload["preprocessing"]["rules"].reverse()
        reversed_config = parse_processing_config(reversed_payload)
        self.assertNotEqual(first.fingerprint, reversed_config.fingerprint)

    def test_rejects_unknown_fields_duplicate_ids_and_missing_files(self) -> None:
        payload = tomllib.loads(self.example_path.read_text(encoding="utf-8"))
        payload["unexpected"] = True
        with self.assertRaisesRegex(ConfigurationError, "未知字段"):
            parse_processing_config(payload)

        payload = tomllib.loads(self.example_path.read_text(encoding="utf-8"))
        payload["preprocessing"]["rules"][1]["id"] = payload["preprocessing"]["rules"][0]["id"]
        with self.assertRaisesRegex(ConfigurationError, "重复"):
            parse_processing_config(payload)

        with self.assertRaisesRegex(ConfigurationError, "不存在"):
            load_processing_config(self.example_path.with_name("missing-processing.toml"))

    def test_rejects_bad_capture_references_even_when_rule_is_disabled(self) -> None:
        self.payload["preprocessing"]["rules"][1]["enabled"] = False
        self.payload["preprocessing"]["rules"][1]["replacement"] = r"E\g<missing>"
        with self.assertRaisesRegex(ConfigurationError, "不存在的捕获组"):
            parse_processing_config(self.payload)

    def test_rejects_unsupported_result_processing(self) -> None:
        self.payload["result_processing"]["enabled"] = True
        with self.assertRaisesRegex(ConfigurationError, "只预留透传入口"):
            parse_processing_config(self.payload)
