"""实体级BIO评测指标。"""

from __future__ import annotations

from typing import Sequence

from anitopy_ml.errors import SchemaValidationError
from anitopy_ml.modeling.decoder import _parts


def bio_entities(labels: Sequence[str]) -> set[tuple[int, int, str]]:
    """将合法BIO序列转换为实体区间集合。"""
    result: set[tuple[int, int, str]] = set()
    start: int | None = None
    entity_type: str | None = None
    for index, label in enumerate((*labels, "O")):
        prefix, current_type = _parts(label)
        continuation = prefix == "I" and start is not None and current_type == entity_type
        if continuation:
            continue
        if start is not None:
            result.add((start, index, entity_type or ""))
            start, entity_type = None, None
        if prefix in {"B", "I"}:
            start, entity_type = index, current_type
    return result


def entity_metrics(predicted: Sequence[str], expected: Sequence[str]) -> dict[str, float | int]:
    """计算严格实体边界匹配的精确率、召回率和F1。"""
    if len(predicted) != len(expected):
        raise SchemaValidationError("预测标签和真实标签长度不一致。")
    predicted_entities = bio_entities(predicted)
    expected_entities = bio_entities(expected)
    true_positive = len(predicted_entities & expected_entities)
    precision = true_positive / len(predicted_entities) if predicted_entities else 0.0
    recall = true_positive / len(expected_entities) if expected_entities else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "预测实体数": len(predicted_entities),
        "真实实体数": len(expected_entities),
        "正确实体数": true_positive,
        "精确率": precision,
        "召回率": recall,
        "F1": f1,
    }
