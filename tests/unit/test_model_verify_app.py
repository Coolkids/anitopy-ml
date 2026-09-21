"""本地模型验证页面测试。"""

import runpy
import unittest

from anitopy_ml.schemas import ExtractedFields, Evidence, ParseResult, Span


class ModelVerifyAppTests(unittest.TestCase):
    """验证页面突出展示核心字段且所有标签为中文。"""

    def test_result_page_displays_core_fields_and_chinese_labels(self) -> None:
        namespace = runpy.run_path("tools/model_verify_app.py")
        result = ParseResult(
            raw_text="[字幕组] 示例作品 第二季 07",
            extracted=ExtractedFields(
                title="示例作品",
                title_aliases=["Example Work"],
                seasons=[2],
                episodes=[{"raw": "07", "value": "7", "numbering": "unknown"}],
                release_groups=["字幕组"],
            ),
            evidence={
                "title": Evidence(
                    spans=(Span(6, 10, "示例作品", "TITLE", source="model"),),
                    source="model",
                )
            },
            model_version="测试模型",
        )
        content = namespace["render_verification_page"](result=result).decode("utf-8")
        self.assertIn("核心识别结果", content)
        self.assertIn("主标题", content)
        self.assertIn("作品别名", content)
        self.assertIn("季数", content)
        self.assertIn("集数", content)
        self.assertIn("7（原文：07）", content)
        self.assertIn("作品主标题", content)
        self.assertNotIn(">TITLE<", content)

    def test_empty_page_contains_title_input(self) -> None:
        namespace = runpy.run_path("tools/model_verify_app.py")
        content = namespace["render_verification_page"]().decode("utf-8")
        self.assertIn("待验证名称", content)
        self.assertIn("开始识别", content)
        self.assertIn("标题、别名、季数、集数", content)
