"""对已导入标题批量运行冻结旧解析器。"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from anitopy_ml.data.ingest import TitleRecord
from anitopy_ml.errors import InputValidationError
from anitopy_ml.legacy import load_legacy_parse


def read_jsonl_records(path: str | Path) -> Iterable[TitleRecord]:
    """按行读取由导入命令生成的JSONL，并验证必要字段。"""
    source = Path(path)
    if not source.is_file():
        raise InputValidationError(f"未找到输入JSONL文件：{source}")
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise InputValidationError(f"JSONL第{line_number}行为空。")
            try:
                payload = json.loads(line)
                yield TitleRecord(
                    sample_id=payload["sample_id"],
                    source_row=payload["source_row"],
                    raw_text=payload["raw_text"],
                    source_sha256=payload["source_sha256"],
                    raw_text_sha256=payload["raw_text_sha256"],
                )
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise InputValidationError(f"JSONL第{line_number}行结构无效。") from error


def write_legacy_baseline(
    records: Iterable[TitleRecord],
    *,
    legacy_source: str | Path,
    output: str | Path,
) -> dict[str, int]:
    """逐条解析标题并写出可与原始样本关联的JSONL结果。"""
    parser = load_legacy_parse(legacy_source)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    exception_count = 0
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            total += 1
            payload: dict[str, Any] = {
                "sample_id": record.sample_id,
                "source_row": record.source_row,
                "raw_text_sha256": record.raw_text_sha256,
            }
            try:
                payload["legacy_result"] = parser(record.raw_text, options={})
            except Exception as error:  # 旧实现错误需要进入报告，不能静默丢弃样本
                exception_count += 1
                payload["legacy_error"] = repr(error)
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
    return {"total": total, "exception_count": exception_count}
