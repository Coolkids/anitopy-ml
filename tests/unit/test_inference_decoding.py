"""离线推理结果组装测试。"""

import unittest

from anitopy_ml.errors import SchemaValidationError
from anitopy_ml.inference.decoding import build_parse_result, spans_from_bio


class InferenceDecodingTests(unittest.TestCase):
    """验证预测标签能够稳定还原到原始标题。"""

    def test_build_result_preserves_title_evidence_and_rule_fields(self) -> None:
        raw_text = "示例 S01E02 1080p"
        labels = ["B-TITLE", "I-TITLE", "O", "B-SEASON_EXPR", "I-SEASON_EXPR"] + ["O"] * 10
        result = build_parse_result(raw_text, labels, confidences=[0.8] * len(labels), model_version="测试模型")
        self.assertEqual(result.extracted.title, "示例")
        self.assertEqual(result.extracted.seasons, [1])
        self.assertEqual(result.evidence["title"].spans[0].start, 0)
        self.assertFalse(result.evidence["title"].calibrated)

    def test_illegal_inside_label_is_rejected(self) -> None:
        with self.assertRaises(SchemaValidationError):
            spans_from_bio("示例", ["I-TITLE", "O"])

    def test_build_result_marks_transformed_confidence_as_calibrated(self) -> None:
        result = build_parse_result(
            "示例",
            ["B-TITLE", "I-TITLE"],
            confidences=[0.8, 0.8],
            confidence_transform=lambda _label, _score: (0.75, True),
        )
        self.assertEqual(result.evidence["title"].confidence, 0.75)
        self.assertTrue(result.evidence["title"].calibrated)
