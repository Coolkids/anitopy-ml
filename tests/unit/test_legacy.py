"""旧解析器兼容层测试。"""

import unittest
from pathlib import Path
from unittest.mock import patch

from anitopy_ml.errors import ConfigurationError
from anitopy_ml.legacy import _fixture_options, parse_legacy


class LegacyTests(unittest.TestCase):
    """使用最小旧包验证显式加载和选项复制。"""

    def test_options_are_copied(self) -> None:
        def fake_parse(title: str, options: dict[str, object]) -> dict[str, object]:
            options.setdefault("旧默认项", True)
            return {"file_name": title, "选项数量": len(options)}

        options = {"parse_episode_number": True}
        with patch("anitopy_ml.legacy.load_legacy_parse", return_value=fake_parse):
            result = parse_legacy("示例作品 - 01", source="不实际读取的路径", options=options)
        self.assertEqual(options, {"parse_episode_number": True})
        self.assertEqual(result, {"file_name": "示例作品 - 01", "选项数量": 2})

    def test_source_requires_license(self) -> None:
        with self.assertRaises(ConfigurationError):
            parse_legacy("示例作品", source=Path("不存在的旧解析器目录"))

    def test_fixture_options_are_converted(self) -> None:
        options = _fixture_options({"option_parse_episode_number": False})
        self.assertEqual(options, {"parse_episode_number": False})
