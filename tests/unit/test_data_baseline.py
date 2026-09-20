"""批量旧解析结果的测试。"""

from __future__ import annotations

import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from anitopy_ml.data.baseline import read_jsonl_records, write_legacy_baseline
from anitopy_ml.data.ingest import read_title_rows


class _NonClosingStringIO(io.StringIO):
    """供上下文管理器使用、但可在断言时读取内容的内存流。"""

    def close(self) -> None:
        """保留内容到测试断言结束。"""


class DataBaselineTests(unittest.TestCase):
    """确保每条旧解析结果均可回溯到导入样本。"""

    def test_writes_linked_results_and_records_exceptions(self) -> None:
        records = read_title_rows(
            io.StringIO("record_title\n可解析标题\n异常标题\n"),
            column="record_title",
            source_sha256="e" * 64,
        )

        def fake_parse(title: str, *, options: dict[str, object]) -> dict[str, str]:
            if title == "异常标题":
                raise ValueError("示例错误")
            return {"title": title}

        stream = _NonClosingStringIO()
        with (
            patch("anitopy_ml.data.baseline.load_legacy_parse", return_value=fake_parse),
            patch.object(Path, "open", return_value=stream),
        ):
            report = write_legacy_baseline(records, legacy_source="旧项目", output="基线.jsonl")
        payloads = [json.loads(line) for line in stream.getvalue().splitlines()]

        self.assertEqual(report, {"total": 2, "exception_count": 1})
        self.assertEqual(payloads[0]["legacy_result"], {"title": "可解析标题"})
        self.assertEqual(payloads[1]["legacy_error"], "ValueError('示例错误')")
        self.assertEqual(payloads[0]["sample_id"], records[0].sample_id)

    def test_reads_jsonl_records(self) -> None:
        line = json.dumps(
            {
                "sample_id": "样本",
                "source_row": 2,
                "raw_text": "示例作品",
                "source_sha256": "a" * 64,
                "raw_text_sha256": "b" * 64,
            },
            ensure_ascii=False,
        )
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "open", return_value=io.StringIO(line + "\n")),
        ):
            records = list(read_jsonl_records("输入.jsonl"))
        self.assertEqual(records[0].raw_text, "示例作品")
