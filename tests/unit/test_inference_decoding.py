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

    def test_model_spans_fill_release_group_and_numeric_season_episode_fields(self) -> None:
        raw_text = "【字幕组】★示例作品★第二季★07★12(完)★1920x1080"
        labels = ["O"] * len(raw_text)

        def mark(text: str, label: str) -> None:
            start = raw_text.index(text)
            labels[start] = f"B-{label}"
            for index in range(start + 1, start + len(text)):
                labels[index] = f"I-{label}"

        mark("【字幕组】", "RELEASE_GROUP")
        mark("第二季", "SEASON_EXPR")
        mark("07", "EPISODE_EXPR")
        mark("12(完)", "EPISODE_COUNT_EXPR")
        mark("1920x1080", "RESOLUTION")

        result = build_parse_result(raw_text, labels, confidences=[0.9] * len(labels))

        self.assertEqual(result.extracted.release_groups, ["字幕组"])
        self.assertEqual(result.extracted.seasons, [2])
        self.assertEqual(result.extracted.episodes, [{"raw": "07", "value": "7", "numbering": "unknown"}])
        self.assertEqual(result.extracted.declared_episode_count, 12)
        self.assertEqual(result.extracted.resolution, ["1920x1080"])
