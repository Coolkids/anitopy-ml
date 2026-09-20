"""训练对齐问题的可审计输出。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def write_alignment_issues(
    issues: Iterable[dict[str, object]],
    output: str | Path,
) -> int:
    """将无法对齐的样本与原因写为UTF-8 JSONL，不修改原始标注。"""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for item in issues:
            if not item.get("sample_id") or not item.get("reason"):
                raise ValueError("对齐问题记录必须包含样本ID和原因。")
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
            count += 1
    return count
