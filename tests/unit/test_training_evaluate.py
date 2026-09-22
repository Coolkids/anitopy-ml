"""冻结测试评测聚合测试。"""

import unittest
from unittest.mock import patch

from anitopy_ml.schemas import Evidence, ExtractedFields, ParseResult, Span
from anitopy_ml.training.evaluate import evaluate_records, evaluate_validation_extractor


class _FixedParser:
    """返回确定预测，避免单元测试加载模型权重。"""

    def parse(self, title: str) -> ParseResult:
        return ParseResult(
            raw_text=title,
            evidence={
                "title": Evidence(
                    spans=(Span(0, 2, "示例", "TITLE", source="model"),),
                    source="model",
                    confidence=0.9,
                    calibrated=True,
                )
            },
        )


class EvaluationTests(unittest.TestCase):
    """验证严格边界指标和字段统计。"""

    def test_reports_exact_entity_and_full_title_metrics(self) -> None:
        report = evaluate_records(
            [{"sample_id": "一", "text": "示例", "labels": ["B-TITLE", "I-TITLE"]}],
            _FixedParser(),
        )
        self.assertEqual(report["实体指标"]["F1"], 1.0)
        self.assertEqual(report["整条严格完全正确率"], 1.0)
        self.assertEqual(report["核心字段指标"]["名称与别名联合"]["完全正确率"], 1.0)
        self.assertEqual(report["核心字段整条完全正确率"], 1.0)
        self.assertEqual(report["接受策略统计"]["自动接收"], 1)

    def test_name_and_alias_may_be_recognized_as_one_combined_boundary(self) -> None:
        class CombinedNameParser:
            def parse(self, title: str) -> ParseResult:
                return ParseResult(
                    raw_text=title,
                    evidence={
                        "title": Evidence(
                            spans=(Span(0, len(title), title, "TITLE", source="model"),),
                            source="model",
                        )
                    },
                )

        report = evaluate_records(
            [
                {
                    "sample_id": "名称合并",
                    "text": "中文名 / English Name",
                    "labels": [
                        "B-TITLE", "I-TITLE", "I-TITLE", "O",
                        "O", "O", "B-TITLE_ALIAS", *["I-TITLE_ALIAS"] * 11,
                    ],
                }
            ],
            CombinedNameParser(),
        )
        self.assertEqual(report["整条严格完全正确率"], 0.0)
        self.assertEqual(report["核心字段指标"]["名称与别名联合"]["完全正确率"], 1.0)

    def test_season_and_episode_are_compared_by_numeric_meaning(self) -> None:
        class NumericParser:
            def parse(self, title: str) -> ParseResult:
                return ParseResult(
                    raw_text=title,
                    extracted=ExtractedFields(
                        seasons=[1],
                        episodes=[{"raw": "02", "value": "2", "numbering": "season"}],
                    ),
                )

        report = evaluate_records(
            [
                {
                    "sample_id": "数值语义",
                    "text": "示例 S01E02",
                    "labels": [
                        "B-TITLE", "I-TITLE", "O", "B-SEASON_EXPR", "I-SEASON_EXPR", "I-SEASON_EXPR",
                        "B-EPISODE_EXPR", "I-EPISODE_EXPR", "I-EPISODE_EXPR",
                    ],
                }
            ],
            NumericParser(),
        )
        self.assertEqual(report["核心字段指标"]["季数"]["完全正确率"], 1.0)
        self.assertEqual(report["核心字段指标"]["集数"]["完全正确率"], 1.0)

    def test_declared_episode_range_includes_its_total_count(self) -> None:
        class CollectionParser:
            def parse(self, title: str) -> ParseResult:
                return ParseResult(
                    raw_text=title,
                    extracted=ExtractedFields(
                        episode_ranges=[{"raw": "01-12", "start": "1", "end": "12", "numbering": "unknown"}],
                        declared_episode_count=12,
                    ),
                )

        report = evaluate_records(
            [{"sample_id": "合集", "text": "01-12 END", "labels": ["B-EPISODE_COUNT_EXPR", *["I-EPISODE_COUNT_EXPR"] * 8]}],
            CollectionParser(),
        )
        self.assertEqual(report["核心字段指标"]["集数"]["完全正确率"], 1.0)

    def test_validation_evaluation_writes_validation_metadata(self) -> None:
        records = [{"sample_id": "验证", "text": "示例", "labels": ["B-TITLE", "I-TITLE"]}]
        with (
            patch("anitopy_ml.training.evaluate._read_records", return_value=records),
            patch("anitopy_ml.training.evaluate.MediaParser.from_pretrained", return_value=_FixedParser()),
            patch("anitopy_ml.training.evaluate.Path.mkdir"),
            patch("anitopy_ml.training.evaluate.Path.write_text"),
        ):
            report = evaluate_validation_extractor(
                model_directory="候选模型", validation_path="验证.jsonl", output_path="报告.json", device="cpu"
            )
        self.assertIn("验证集", str(report["说明"]))
        self.assertEqual(report["验证样本数"], 1)
        self.assertNotIn("测试样本数", report)
        self.assertEqual(report["验证文件"], "验证.jsonl")
