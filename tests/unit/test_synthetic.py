"""名称模板合成训练数据测试。"""

import unittest
import re

from anitopy_ml.data.synthetic import audit_synthetic_records, deduplicate_synthetic_records, generate_synthetic_records, split_synthetic_records
from anitopy_ml.errors import SchemaValidationError


def template_payload() -> dict[str, object]:
    """构造覆盖标题别名、季数和中英文字幕字段的最小模板数据。"""
    return {
        "title": [{"title": "示例作品 第二季", "title_alias": ["Demo", "演示"], "season_expr": ["第二季"]}],
        "episode_expr": ["EP01"],
        "episode_count_expr": ["全12集"],
        "release_group": ["字幕组"],
        "source": ["WEB-DL"],
        "resolution": ["1080p"],
        "video_term": ["H264"],
        "audio_term": ["AAC"],
        "subtitle_language_chs": ["简繁"],
        "subtitle_language_en": ["CHS&CHT"],
        "subtitle_mode": ["内封"],
        "file_extension": ["mkv"],
        "bit_depth": ["10bit"],
        "name_template": [
            {
                "name": ["[$release_group] $title$title_alias [$episode_expr][&subtitle_language_chs&subtitle_mode].$file_extension"],
                "title_alias_open": "（",
                "title_alias_close": "）",
                "title_alias_separator": " / ",
            }
        ],
    }


class SyntheticDataTests(unittest.TestCase):
    """验证模板占位符、别名循环和字符标签保持一致。"""

    def test_alias_wrappers_and_subtitle_labels_are_generated(self) -> None:
        record = generate_synthetic_records(template_payload(), count=1, seed=1)[0]
        self.assertEqual(record["text"], "[字幕组] 示例作品 第二季（Demo） / （演示） [EP01][简繁内封].mkv")
        spans = record["spans"]
        self.assertTrue(any(item["label"] == "TITLE_ALIAS" and item["text"] == "Demo" for item in spans))
        self.assertTrue(any(item["label"] == "SEASON_EXPR" and item["text"] == "第二季" for item in spans))
        self.assertTrue(any(item["label"] == "SUBTITLE_LANGUAGE" and item["text"] == "简繁" for item in spans))
        self.assertEqual(len(record["characters"]), len(record["labels"]))
        self.assertEqual(record["template_group_id"], "synthetic-template-group-01")

    def test_seed_makes_generation_reproducible(self) -> None:
        first = generate_synthetic_records(template_payload(), count=3, seed=8)
        second = generate_synthetic_records(template_payload(), count=3, seed=8)
        self.assertEqual(first, second)

    def test_random_episode_expression_is_expanded_and_zero_padded(self) -> None:
        payload = template_payload()
        payload["episode_expr"] = ["%random(2)"]
        record = generate_synthetic_records(payload, count=1, seed=3)[0]
        episode_spans = [item for item in record["spans"] if item["label"] == "EPISODE_EXPR"]
        self.assertEqual(len(episode_spans), 1)
        self.assertRegex(str(episode_spans[0]["text"]), re.compile(r"^[0-9]{2}$"))
        self.assertNotEqual(episode_spans[0]["text"], "%random(2)")

    def test_english_title_is_primary_and_chinese_title_is_alias(self) -> None:
        payload = template_payload()
        payload["title"] = [
            {
                "title_en": "Example Work",
                "title_cn": "示例作品",
                "title_alias": [],
                "season_expr": [],
            }
        ]
        payload["name_template"] = [
            {
                "name": ["$title_en / $title_cn [$episode_expr]"],
                "title_alias_open": "",
                "title_alias_close": "",
                "title_alias_separator": "",
            }
        ]
        record = generate_synthetic_records(payload, count=1, seed=17)[0]
        spans = record["spans"]
        self.assertTrue(any(item["label"] == "TITLE" and item["text"] == "Example Work" for item in spans))
        self.assertTrue(any(item["label"] == "TITLE_ALIAS" and item["text"] == "示例作品" for item in spans))
        self.assertEqual(record["work_key"], "example work")

    def test_english_and_chinese_title_fall_back_to_legacy_fields(self) -> None:
        payload = template_payload()
        payload["name_template"] = [
            {
                "name": ["$title_en / $title_cn [$episode_expr]"],
                "title_alias_open": "",
                "title_alias_close": "",
                "title_alias_separator": "",
            }
        ]
        record = generate_synthetic_records(payload, count=1, seed=18)[0]
        self.assertTrue(any(item["label"] == "TITLE" and item["text"] == "Demo" for item in record["spans"]))
        self.assertTrue(
            any(item["label"] == "TITLE_ALIAS" and "示例作品" in item["text"] for item in record["spans"])
        )
        self.assertTrue(any(item["label"] == "SEASON_EXPR" and item["text"] == "第二季" for item in record["spans"]))
        self.assertEqual(record["work_key"], "demo")

    def test_legacy_title_alias_placeholder_is_still_supported(self) -> None:
        payload = template_payload()
        payload["name_template"] = [
            {
                "name": ["$title_as_title_alias_1 [$episode_expr]"],
                "title_alias_open": "",
                "title_alias_close": "",
                "title_alias_separator": "",
            }
        ]
        record = generate_synthetic_records(payload, count=1, seed=19)[0]
        self.assertFalse(any(item["label"] == "TITLE" for item in record["spans"]))
        self.assertTrue(any(item["label"] == "TITLE_ALIAS" for item in record["spans"]))

    def test_adjacent_title_and_alias_without_boundary_is_rejected(self) -> None:
        payload = template_payload()
        payload["name_template"] = [
            {
                "name": ["$title$title_alias [$episode_expr]"],
                "title_alias_open": "",
                "title_alias_close": "",
                "title_alias_separator": " / ",
            }
        ]
        with self.assertRaisesRegex(SchemaValidationError, "缺少明确分隔符"):
            generate_synthetic_records(payload, count=1, seed=21)

    def test_title_alias_begin_provides_title_boundary(self) -> None:
        payload = template_payload()
        payload["name_template"] = [
            {
                "name": ["$title$title_alias [$episode_expr]"],
                "title_alias_open": "",
                "title_alias_begin": " / ",
                "title_alias_close": "",
                "title_alias_separator": " / ",
            }
        ]
        record = generate_synthetic_records(payload, count=1, seed=23)[0]
        self.assertIn(" / Demo / 演示", record["text"])

    def test_multiple_aliases_without_separator_or_wrapper_is_rejected(self) -> None:
        payload = template_payload()
        payload["name_template"] = [
            {
                "name": ["$title_alias [$episode_expr]"],
                "title_alias_open": "",
                "title_alias_close": "",
                "title_alias_separator": "",
            }
        ]
        with self.assertRaisesRegex(SchemaValidationError, "title_alias_separator"):
            generate_synthetic_records(payload, count=1, seed=22)

    def test_each_alias_is_wrapped_when_separator_is_empty(self) -> None:
        payload = template_payload()
        payload["name_template"] = [
            {
                "name": ["$title$title_alias [$episode_expr]"],
                "title_alias_open": "[",
                "title_alias_close": "]",
                "title_alias_separator": "",
                "title_alias_begin": "",
            }
        ]
        record = generate_synthetic_records(payload, count=1, seed=25)[0]
        self.assertIn("[Demo][演示]", record["text"])

    def test_template_season_month_range_and_indexed_alias_are_generated(self) -> None:
        payload = template_payload()
        payload["title"] = [
            {"title": "旧作 第二季", "title_alias": ["Old Work"], "season_expr": ["第二季"]},
            {"title": "新作", "title_alias": ["New Work", "第二别名"], "season_expr": []},
        ]
        payload["name_template"] = [
            {
                "name": ["[$release_group][$random_month月新番]$title_alias[1] $season_exprE$episode_expr"],
                "exclude_title_season_expr": True,
                "title_alias_open": "",
                "title_alias_close": "",
                "title_alias_separator": "",
                "random_month": ["1"],
                "season_expr": ["S0%random_range(1-6)"],
            }
        ]
        record = generate_synthetic_records(payload, count=1, seed=4)[0]
        self.assertIn("[1月新番]New Work S0", record["text"])
        self.assertNotIn("第二别名", record["text"])
        self.assertNotIn("旧作", record["text"])
        self.assertNotIn("%random", record["text"])
        self.assertTrue(any(item["label"] == "TITLE_ALIAS" and item["text"] == "New Work" for item in record["spans"]))
        self.assertTrue(any(item["label"] == "SEASON_EXPR" and re.fullmatch(r"S0[1-6]", str(item["text"])) for item in record["spans"]))

    def test_template_local_noise_is_labeled(self) -> None:
        payload = template_payload()
        payload["name_template"] = [
            {
                "name": ["[$release_group][$noise][$title][$episode_expr]"],
                "title_alias_open": "",
                "title_alias_close": "",
                "title_alias_separator": "",
                "noise": ["4月新番"],
            }
        ]
        record = generate_synthetic_records(payload, count=1, seed=20)[0]
        self.assertTrue(any(item["label"] == "NOISE" and item["text"] == "4月新番" for item in record["spans"]))
        noise_start = record["text"].index("4月新番")
        self.assertEqual(record["labels"][noise_start], "B-NOISE")

    def test_noise2_and_random_string_are_labeled(self) -> None:
        payload = template_payload()
        payload["name_template"] = [
            {
                "name": ["[$noise][$noise2]$title"],
                "title_alias_open": "",
                "title_alias_close": "",
                "title_alias_separator": " / ",
                "noise": ["检索:%random_str($title, 3)"],
                "noise2": ["检索用:%random_str($title, 3)"],
            }
        ]
        first = generate_synthetic_records(payload, count=1, seed=24)[0]
        second = generate_synthetic_records(payload, count=1, seed=24)[0]
        self.assertEqual(first, second)
        noise_spans = [item for item in first["spans"] if item["label"] == "NOISE"]
        self.assertEqual(len(noise_spans), 2)
        for span in noise_spans:
            prefix, fragment = str(span["text"]).split(":", maxsplit=1)
            self.assertIn(prefix, {"检索", "检索用"})
            self.assertGreaterEqual(len(fragment), 1)
            self.assertLessEqual(len(fragment), 3)
            self.assertIn(fragment, "示例作品 第二季")

    def test_release_version_is_labeled_from_template_local_values(self) -> None:
        payload = template_payload()
        payload["name_template"] = [
            {
                "name": ["$title $episode_expr$release_version"],
                "title_alias_open": "",
                "title_alias_close": "",
                "title_alias_separator": " / ",
                "release_version": ["v%random(1)"],
            }
        ]
        record = generate_synthetic_records(payload, count=1, seed=31)[0]
        release_version = [item for item in record["spans"] if item["label"] == "RELEASE_VERSION"]
        self.assertEqual(len(release_version), 1)
        self.assertRegex(str(release_version[0]["text"]), r"^v[1-9]$")

    def test_template_disable_condition_filters_titles_by_alias_count(self) -> None:
        payload = template_payload()
        payload["title"] = [
            {"title": "单别名", "title_alias": ["One"], "season_expr": []},
            {"title": "三别名", "title_alias": ["One", "Two", "Three"], "season_expr": []},
        ]
        payload["name_template"] = [
            {
                "name": ["$title$title_alias [$episode_expr]"],
                "disable": ["%len($title_alias) > 2"],
                "title_alias_open": "",
                "title_alias_close": "",
                "title_alias_separator": " / ",
                "title_alias_begin": " / ",
            }
        ]
        records = generate_synthetic_records(payload, count=12, seed=32)
        self.assertTrue(all("三别名" not in str(record["text"]) for record in records))

    def test_season_titles_are_sampled_at_the_requested_rate(self) -> None:
        payload = template_payload()
        payload["title"] = [
            {"title": "有季数作品 第二季", "title_alias": [], "season_expr": ["第二季"]},
            {"title": "普通作品", "title_alias": [], "season_expr": []},
        ]
        records = generate_synthetic_records(payload, count=200, seed=13, season_rate=0.7)
        season_count = sum(
            any(item["label"] == "SEASON_EXPR" for item in record["spans"])
            for record in records
        )
        self.assertGreaterEqual(season_count, 120)

    def test_audit_reports_duplicate_and_bio_consistency(self) -> None:
        records = generate_synthetic_records(template_payload(), count=1, seed=8)
        report = audit_synthetic_records([*records, *records])
        self.assertEqual(report["重复样本ID数"], 1)
        self.assertEqual(report["BIO不一致样本数"], 0)

    def test_deduplicate_removes_same_text_and_labels(self) -> None:
        records = generate_synthetic_records(template_payload(), count=1, seed=8)
        duplicate = dict(records[0])
        duplicate["sample_id"] = "另一个样本ID"
        kept, report = deduplicate_synthetic_records([records[0], duplicate])
        self.assertEqual(len(kept), 1)
        self.assertEqual(report["重复文本数"], 1)

    def test_title_and_template_group_never_cross_synthetic_splits(self) -> None:
        records = [
            {
                "sample_id": f"synthetic-{title_index}-{template_index}",
                "template_group_id": f"模板组{template_index}",
                "template": f"模板{template_index}",
                "spans": [{"label": "TITLE", "text": f"作品{title_index}"}],
            }
            for title_index in range(100)
            for template_index in range(3)
        ]
        splits, report = split_synthetic_records(records, seed=9)
        title_splits: dict[str, set[str]] = {}
        template_splits: dict[str, set[str]] = {}
        for split, values in splits.items():
            for record in values:
                title = record["spans"][0]["text"]
                title_splits.setdefault(title, set()).add(split)
                template_splits.setdefault(record["template_group_id"], set()).add(split)
        self.assertTrue(all(len(values) == 1 for values in title_splits.values()))
        self.assertTrue(all(len(values) == 1 for values in template_splits.values()))
        self.assertGreater(report["双隔离排除样本数"], 0)
        self.assertEqual(report["模板组数"], 3)

    def test_alias_only_template_uses_explicit_work_key_for_split(self) -> None:
        records = [
            {
                "sample_id": f"别名样本-{work_index}-{template_index}",
                "work_key": f"作品{work_index}",
                "template_group_id": f"模板组{template_index}",
                "template": f"模板{template_index}",
                "spans": [{"label": "TITLE_ALIAS", "text": f"Alias {work_index}"}],
            }
            for work_index in range(10)
            for template_index in range(3)
        ]
        splits, _ = split_synthetic_records(records, seed=11)
        self.assertEqual(sum(len(items) for items in splits.values()), 10)

    def test_work_assignments_are_balanced_before_template_intersection(self) -> None:
        records = [
            {
                "sample_id": f"样本-{title_index}-{template_index}",
                "template_group_id": f"模板组{template_index}",
                "template": f"模板{template_index}",
                "spans": [{"label": "TITLE", "text": f"作品{title_index}"}],
            }
            for title_index in range(10)
            for template_index in range(3)
        ]
        _, report = split_synthetic_records(records, seed=10)
        self.assertEqual(report["分区作品数"], {"train": 8, "validation": 1, "test": 1})
