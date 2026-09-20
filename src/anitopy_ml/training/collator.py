"""训练批次的动态填充，不绑定具体深度学习框架。"""

from __future__ import annotations

from typing import Any, Sequence

from anitopy_ml.errors import SchemaValidationError
from anitopy_ml.training.alignment import IGNORE_LABEL_ID


def dynamic_pad(
    batch: Sequence[dict[str, Any]],
    *,
    pad_token_id: int,
) -> dict[str, list[list[int]]]:
    """按当前批次最长序列填充输入、注意力、标签和损失掩码。"""
    if not batch:
        raise SchemaValidationError("动态填充不能处理空批次。")
    required = ("input_ids", "attention_mask", "labels", "loss_mask")
    for item in batch:
        if any(name not in item for name in required):
            raise SchemaValidationError("训练批次缺少必要序列字段。")
        lengths = {len(item[name]) for name in required}
        if len(lengths) != 1:
            raise SchemaValidationError("单个样本的输入、标签和掩码长度不一致。")
    target_length = max(len(item["input_ids"]) for item in batch)
    result = {"input_ids": [], "attention_mask": [], "labels": [], "loss_mask": []}
    for item in batch:
        length = len(item["input_ids"])
        padding = target_length - length
        result["input_ids"].append([int(value) for value in item["input_ids"]] + [pad_token_id] * padding)
        result["attention_mask"].append([int(value) for value in item["attention_mask"]] + [0] * padding)
        result["labels"].append([int(value) for value in item["labels"]] + [IGNORE_LABEL_ID] * padding)
        result["loss_mask"].append([int(bool(value)) for value in item["loss_mask"]] + [0] * padding)
    return result
