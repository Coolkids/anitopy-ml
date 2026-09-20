"""检查点元数据的原子读写。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anitopy_ml.errors import SchemaValidationError
from anitopy_ml.training.control import TrainingProgress


def write_checkpoint_metadata(
    directory: str | Path,
    *,
    progress: TrainingProgress,
    model_config: dict[str, Any],
) -> Path:
    """原子写入训练恢复元数据；权重和优化器由训练器同目录保存。"""
    progress.validate()
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "checkpoint_metadata.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {"版本": 1, "训练进度": progress.model_dump(), "模型配置": model_config},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def read_checkpoint_metadata(directory: str | Path) -> dict[str, Any]:
    """读取并验证检查点元数据的最低必要字段。"""
    target = Path(directory) / "checkpoint_metadata.json"
    if not target.is_file():
        raise SchemaValidationError("未找到检查点元数据文件。")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        progress = TrainingProgress(**payload["训练进度"])
        progress.validate()
        if not isinstance(payload["模型配置"], dict):
            raise TypeError
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise SchemaValidationError("检查点元数据结构无效。") from error
    return payload
