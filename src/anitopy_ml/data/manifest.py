"""原始标题集的数据清单。"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from anitopy_ml.data.ingest import TitleRecord, profile_records
from anitopy_ml.errors import InputValidationError


def build_manifest(
    records: Iterable[TitleRecord],
    *,
    source_path: str | Path,
    title_column: str,
) -> dict[str, object]:
    """构造可复核的导入清单，不复制原始标题正文。"""
    items = list(records)
    if not items:
        raise InputValidationError("不能为没有记录的数据集生成清单。")

    source_hashes = {record.source_sha256 for record in items}
    if len(source_hashes) != 1:
        raise InputValidationError("同一数据清单只能包含一个源文件哈希。")

    profile = profile_records(items)
    return {
        "清单版本": 1,
        "源文件": str(Path(source_path)),
        "标题列": title_column,
        "源文件SHA256": next(iter(source_hashes)),
        "记录数": len(items),
        "精确重复数": profile["重复记录数"],
        "唯一标题数": profile["唯一标题数"],
        "长度": profile["长度"],
        "模式计数": profile["模式计数"],
        "异常记录": [],
        "说明": "清单只记录统计与哈希；原始标题保留在对应JSONL文件中。",
    }


def write_manifest(manifest: dict[str, object], output: str | Path) -> None:
    """以UTF-8 JSON写入数据清单。"""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
