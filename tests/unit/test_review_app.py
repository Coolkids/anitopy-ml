"""本地审核页面显示文本测试。"""

import runpy
import unittest

from anitopy_ml.annotation.seed import build_seed_record
from anitopy_ml.schemas import Evidence, ParseResult, Span


class ReviewAppTests(unittest.TestCase):
    """验证页面将内部标签转换为中文。"""

    def test_label_name_and_sample_page_are_chinese(self) -> None:
        namespace = runpy.run_path("tools/review_app.py")
        record = build_seed_record("样本-页面", "示例作品 S01E03")
        content = namespace["render_sample"](record.model_dump()).decode("utf-8")
        self.assertEqual(namespace["label_name"]("TITLE"), "作品主标题")
        self.assertIn("季数表达", content)
        self.assertNotIn(">SEASON_EXPR<", content)

    def test_representatives_keep_one_shortest_title_per_loose_key(self) -> None:
        namespace = runpy.run_path("tools/review_app.py")
        entries = [
            {"sample_id": "较长", "raw_text": "[字幕组] 示例作品 EP01 1080p", "review_status": "pending"},
            {"sample_id": "较短", "raw_text": "[发布组] 示例作品 EP02", "review_status": "pending"},
            {"sample_id": "其他", "raw_text": "另一部作品 EP01", "review_status": "pending"},
        ]
        selected = namespace["select_group_representatives"](entries)
        selected_ids = {item["sample_id"] for item in selected}
        self.assertEqual(selected_ids, {"较短", "其他"})
        self.assertEqual(next(item for item in selected if item["sample_id"] == "较短")["相似待审数"], 2)

    def test_model_suggestion_is_displayed_but_not_saved_automatically(self) -> None:
        namespace = runpy.run_path("tools/review_app.py")
        record = build_seed_record("样本-建议", "示例作品 S01E03")
        suggestion = {
            "spans": [{"start": 0, "end": 4, "text": "示例作品", "label": "TITLE", "source": "model"}],
            "extracted": record.model_dump()["extracted"],
            "status": "partial",
            "warnings": ["模型尚未完成校准。"],
            "model_version": "测试模型",
        }
        content = namespace["render_sample"](
            record.model_dump(),
            model_suggestion=suggestion,
        ).decode("utf-8")
        self.assertIn("模型建议不会自动保存", content)
        self.assertIn("将模型建议载入编辑草稿", content)
        self.assertIn("作品主标题：[0, 4)「示例作品」", content)

    def test_build_model_suggestion_only_collects_model_evidence(self) -> None:
        namespace = runpy.run_path("tools/review_app.py")
        result = ParseResult(
            raw_text="示例作品 S01",
            evidence={
                "title": Evidence(
                    spans=(Span(0, 4, "示例作品", "TITLE", source="model"),),
                    source="model",
                ),
                "season_expr": Evidence(
                    spans=(Span(5, 8, "S01", "SEASON_EXPR", source="rule"),),
                    source="rule",
                ),
            },
        )
        suggestion = namespace["build_model_suggestion"](result)
        self.assertEqual(suggestion["spans"], [{"start": 0, "end": 4, "text": "示例作品", "label": "TITLE", "source": "model"}])
