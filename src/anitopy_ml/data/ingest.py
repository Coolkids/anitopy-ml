"""CSV标题数据的无损导入和可重复统计。"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, TextIO

from anitopy_ml.errors import InputValidationError


@dataclass(frozen=True, slots=True)
class TitleRecord:
    """从CSV导入的一条原始标题记录。"""

    sample_id: str
    source_row: int
    raw_text: str
    source_sha256: str
    raw_text_sha256: str


def sha256_file(path: Path) -> str:
    """分块计算文件SHA-256，避免将整个文件读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_title_rows(stream: TextIO, *, column: str, source_sha256: str) -> list[TitleRecord]:
    """读取已打开的UTF-8 CSV流，并验证指定标题列。"""
    reader = csv.DictReader(stream)
    if reader.fieldnames is None or column not in reader.fieldnames:
        raise InputValidationError(f"CSV中未找到标题列“{column}”。")

    records: list[TitleRecord] = []
    for row_number, row in enumerate(reader, start=2):
        title = row.get(column)
        if title is None or not title.strip():
            raise InputValidationError(f"CSV第{row_number}条记录的标题为空。")
        raw_hash = hashlib.sha256(title.encode("utf-8")).hexdigest()
        sample_key = f"{source_sha256}:{row_number}:{raw_hash}"
        sample_id = hashlib.sha256(sample_key.encode("utf-8")).hexdigest()[:24]
        records.append(
            TitleRecord(
                sample_id=sample_id,
                source_row=row_number,
                raw_text=title,
                source_sha256=source_sha256,
                raw_text_sha256=raw_hash,
            )
        )
    return records


def ingest_csv(path: str | Path, *, column: str = "record_title") -> list[TitleRecord]:
    """以UTF-8或UTF-8 BOM编码导入标题CSV。"""
    source = Path(path)
    if not source.is_file():
        raise InputValidationError(f"未找到输入文件：{source}")
    file_hash = sha256_file(source)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        return read_title_rows(stream, column=column, source_sha256=file_hash)


def write_jsonl(records: Iterable[TitleRecord], output: str | Path) -> int:
    """将原始记录以UTF-8 JSONL写入指定文件。"""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True))
            stream.write("\n")
            count += 1
    return count


def profile_records(records: Iterable[TitleRecord]) -> dict[str, object]:
    """生成标题长度、重复和主要格式模式的统计结果。"""
    items = list(records)
    titles = [item.raw_text for item in items]
    lengths = sorted(map(len, titles))
    patterns = {
        "含中日韩字符": r"[\u4e00-\u9fff\u3040-\u30ff]",
        "含英文字母": r"[A-Za-z]",
        "季集格式": r"(?i)\bS\d{1,2}E\d+",
        "中文季集表达": r"第.{1,5}[季集话話]|全\s*\d+\s*[集话話]",
        "常见分辨率": r"(?i)\b(?:720p|1080[pi]|2160p|4k|8k)\b",
        "独立四位年份": r"\b(?:19|20)\d{2}\b",
        "网页标签或网址": r"(?i)https?://|\[(?:img|url|size|b)\b",
    }
    if not titles:
        raise InputValidationError("没有可统计的标题记录。")
    counts = Counter(titles)
    return {
        "记录数": len(items),
        "唯一标题数": len(counts),
        "重复记录数": sum(count - 1 for count in counts.values()),
        "长度": {
            "最短": lengths[0],
            "中位数": median(lengths),
            "P95": lengths[min(len(lengths) - 1, int(len(lengths) * 0.95))],
            "最长": lengths[-1],
        },
        "模式计数": {name: sum(bool(re.search(pattern, text)) for text in titles) for name, pattern in patterns.items()},
        "源文件SHA256": items[0].source_sha256,
    }
