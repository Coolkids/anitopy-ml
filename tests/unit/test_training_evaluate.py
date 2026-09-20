"""冻结测试评测聚合测试。"""

import unittest

from anitopy_ml.schemas import Evidence, ParseResult, Span
from anitopy_ml.training.evaluate import evaluate_records


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
        self.assertEqual(report["接受策略统计"]["自动接收"], 1)
