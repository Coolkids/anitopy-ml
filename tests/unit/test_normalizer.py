"""标题规范化及偏移映射测试。"""

import unittest

from anitopy_ml.normalizer import normalize_title


class NormalizerTests(unittest.TestCase):
    """验证规范化不会失去原文位置。"""

    def test_full_width_and_whitespace_map_to_source(self) -> None:
        result = normalize_title("作品　／　S01E03")
        self.assertEqual(result.normalized_text, "作品 / S01E03")
        slash = result.normalized_text.index("/")
        self.assertEqual(result.source_range_for_normalized(slash, slash + 1).start, 3)
        self.assertEqual(result.source_range_for_normalized(slash, slash + 1).end, 4)

    def test_url_and_bbcode_are_removed_from_model_view(self) -> None:
        raw_text = "作品 [img]https://example.test/poster.jpg[/img] 1080p"
        result = normalize_title(raw_text)
        self.assertIn("https://example.test", result.normalized_text)
        self.assertNotIn("https://example.test", result.model_text)
        self.assertEqual(result.model_text, "作品 1080p")
        self.assertEqual(len(result.noise_regions), 1)
        self.assertEqual(result.noise_regions[0].kind, "mixed")

    def test_source_to_normalized_mapping_keeps_non_bmp_character(self) -> None:
        result = normalize_title("😀　作品")
        self.assertEqual(result.normalized_text, "😀 作品")
        self.assertEqual(result.normalized_ranges_for_source(0, 1)[0].start, 0)
        self.assertEqual(result.normalized_ranges_for_source(0, 1)[0].end, 1)

    def test_long_model_text_has_overlapping_windows(self) -> None:
        result = normalize_title("示例" + "a" * 18)
        windows = result.make_model_windows(max_chars=8, stride=2)
        self.assertEqual([(item.model_start, item.model_end) for item in windows], [(0, 8), (6, 14), (12, 20)])
        self.assertEqual(windows[-1].source.end, len(result.raw_text))
