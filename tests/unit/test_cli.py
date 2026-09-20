"""命令行基础行为测试。"""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from anitopy_ml.cli import main
from anitopy_ml.schemas import ParseResult


class CliTests(unittest.TestCase):
    """验证环境检查命令使用中文且不泄露凭证。"""

    def test_doctor_json(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["doctor", "--json"])
        report = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertIn("Python版本", report)
        self.assertIn("可选组件", report)

    def test_help_is_chinese(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main([])
        self.assertEqual(exit_code, 0)
        self.assertIn("多语言媒体标题语义解析工具", output.getvalue())

    def test_review_requires_complete_operation_arguments(self) -> None:
        error = io.StringIO()
        with redirect_stderr(error):
            exit_code = main(["annotate", "review", "--database", ":memory:"])
        self.assertEqual(exit_code, 2)
        self.assertIn("配置无效或缺少必要配置", error.getvalue())

    def test_parse_uses_local_parser_and_prints_json(self) -> None:
        output = io.StringIO()
        fake_parser = type("模拟解析器", (), {"parse": lambda _, title: ParseResult(raw_text=title)})()
        with (
            patch("anitopy_ml.cli.MediaParser.from_pretrained", return_value=fake_parser) as load,
            redirect_stdout(output),
        ):
            exit_code = main(["parse", "示例标题", "--model", "模型目录"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue())["raw_text"], "示例标题")
        load.assert_called_once_with("模型目录", device="auto")
