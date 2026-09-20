"""CSV导入和统计的内存测试。"""

import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from anitopy_ml.data.manifest import build_manifest, write_manifest
from anitopy_ml.data.ingest import profile_records, read_title_rows
from anitopy_ml.errors import InputValidationError


class IngestTests(unittest.TestCase):
    """验证CSV语义而非依赖磁盘临时文件。"""

    def test_csv_quote_and_comma_are_preserved(self) -> None:
        stream = io.StringIO('record_title\n"作品, 第一季 S01E01"\n另一部作品 全12集\n')
        records = read_title_rows(stream, column="record_title", source_sha256="a" * 64)
        self.assertEqual(records[0].raw_text, "作品, 第一季 S01E01")
        self.assertEqual(len(records[0].sample_id), 24)
        report = profile_records(records)
        self.assertEqual(report["记录数"], 2)
        self.assertEqual(report["模式计数"]["季集格式"], 1)

    def test_empty_title_is_rejected(self) -> None:
        stream = io.StringIO("record_title\n   \n")
        with self.assertRaises(InputValidationError):
            read_title_rows(stream, column="record_title", source_sha256="b" * 64)

    def test_missing_column_is_rejected(self) -> None:
        stream = io.StringIO("name\n示例作品\n")
        with self.assertRaises(InputValidationError):
            read_title_rows(stream, column="record_title", source_sha256="c" * 64)

    def test_manifest_contains_hash_and_duplicate_summary(self) -> None:
        records = read_title_rows(
            io.StringIO("record_title\n同一标题\n同一标题\n"),
            column="record_title",
            source_sha256="d" * 64,
        )
        manifest = build_manifest(records, source_path="data.csv", title_column="record_title")
        self.assertEqual(manifest["源文件SHA256"], "d" * 64)
        self.assertEqual(manifest["精确重复数"], 1)
        with patch.object(Path, "write_text") as write_text:
            output = Path("清单.json")
            write_manifest(manifest, output)
        content = write_text.call_args.args[0]
        self.assertEqual(json.loads(content)["记录数"], 2)
