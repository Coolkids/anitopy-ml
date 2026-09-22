"""季集与技术字段约束测试。"""

import unittest

from anitopy_ml.constraints import extract_constraints


class ConstraintTests(unittest.TestCase):
    """验证明确格式优先且不猜测标题数字。"""

    def test_range_declared_count_and_technical_terms(self) -> None:
        result = extract_constraints("29岁作品 - EP01 ~ EP10 全12集 [简/繁] (1080p H.264 AAC)")
        self.assertEqual(result.fields.episode_ranges[0]["start"], "1")
        self.assertEqual(result.fields.episode_ranges[0]["end"], "10")
        self.assertEqual(result.fields.declared_episode_count, 12)
        self.assertEqual(result.fields.subtitle_languages, ["zh-Hans", "zh-Hant"])
        self.assertEqual(result.fields.video_codecs, ["H.264"])
        self.assertEqual(result.fields.audio_codecs, ["AAC"])

    def test_title_number_is_not_an_episode(self) -> None:
        result = extract_constraints("3年Z班银八老师 86 1080p")
        self.assertEqual(result.fields.episodes, [])
        self.assertEqual(result.fields.seasons, [])
        self.assertEqual(result.fields.resolution, ["1080p"])

    def test_season_episode_and_chinese_episode(self) -> None:
        result = extract_constraints("示例作品 S02E03 第十二集")
        self.assertEqual(result.fields.seasons, [2])
        self.assertEqual(result.fields.episodes[0], {"raw": "03", "value": "3", "numbering": "season"})
        self.assertIn({"raw": "十二", "value": "12", "numbering": "unknown"}, result.fields.episodes)
        hints = [(item.label, item.raw) for item in result.hints]
        self.assertIn(("SEASON_EXPR", "S02"), hints)
        self.assertIn(("EPISODE_EXPR", "E03"), hints)

    def test_episode_list_and_decimal_are_not_ranges(self) -> None:
        result = extract_constraints("示例作品 [01&03] EP12.5")
        values = [item["value"] for item in result.fields.episodes]
        self.assertEqual(values, ["1", "3", "12.5"])
        self.assertEqual(result.fields.episode_ranges, [])

    def test_video_bit_depth_is_not_an_episode_range(self) -> None:
        result = extract_constraints("示例作品 [01-12 END][AV1-8bit]")
        self.assertEqual(
            result.fields.episode_ranges,
            [{"raw": "01-12", "start": "1", "end": "12", "numbering": "unknown"}],
        )

    def test_special_type_is_not_mapped_to_a_season(self) -> None:
        result = extract_constraints("示例作品 NCOP SP 1080p")
        self.assertEqual(result.fields.special_type, "NCOP")
        self.assertEqual(result.fields.release_kind, "special")
        self.assertEqual(result.fields.seasons, [])
