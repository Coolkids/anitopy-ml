"""字符级BIO与分词器偏移的严格对齐。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from anitopy_ml.errors import SchemaValidationError

IGNORE_LABEL_ID = -100


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """词元标签、损失掩码和无法对齐的原因。"""

    label_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    issues: tuple[str, ...]


def _entities(character_labels: Sequence[str]) -> list[tuple[int, int, str]]:
    """从字符级BIO恢复连续实体区间，并拒绝不合法序列。"""
    entities: list[tuple[int, int, str]] = []
    start: int | None = None
    entity_type: str | None = None
    for index, label in enumerate((*character_labels, "O")):
        if label == "O":
            if start is not None:
                entities.append((start, index, entity_type or ""))
                start, entity_type = None, None
            continue
        prefix, separator, current_type = label.partition("-")
        if separator != "-" or prefix not in {"B", "I"} or not current_type:
            raise SchemaValidationError("字符级BIO标签格式无效。")
        if prefix == "B" or entity_type != current_type or start is None:
            if start is not None:
                entities.append((start, index, entity_type or ""))
            start, entity_type = index, current_type
    return entities


def align_bio_offsets(
    text: str,
    character_labels: Sequence[str],
    offsets: Sequence[tuple[int, int]],
    label_to_id: dict[str, int],
) -> AlignmentResult:
    """以严格边界规则将字符标签映射给词元，异常边界不参与损失。"""
    if len(text) != len(character_labels):
        raise SchemaValidationError("原文长度与字符级BIO标签长度不一致。")
    if "O" not in label_to_id:
        raise SchemaValidationError("标签映射中缺少O标签。")
    entities = _entities(character_labels)
    label_ids: list[int] = []
    loss_mask: list[bool] = []
    issues: list[str] = []
    for token_index, (start, end) in enumerate(offsets):
        if start == end:
            label_ids.append(IGNORE_LABEL_ID)
            loss_mask.append(False)
            continue
        if not 0 <= start < end <= len(text):
            raise SchemaValidationError("分词器偏移超出原始标题边界。")
        intersecting = [entity for entity in entities if start < entity[1] and end > entity[0]]
        if not intersecting:
            label_ids.append(label_to_id["O"])
            loss_mask.append(True)
            continue
        if len(intersecting) != 1:
            issues.append(f"词元{token_index}跨越多个实体边界。")
            label_ids.append(IGNORE_LABEL_ID)
            loss_mask.append(False)
            continue
        entity_start, entity_end, entity_type = intersecting[0]
        if start < entity_start or end > entity_end:
            issues.append(f"词元{token_index}覆盖实体边界之外的字符。")
            label_ids.append(IGNORE_LABEL_ID)
            loss_mask.append(False)
            continue
        label = f"B-{entity_type}" if start == entity_start else f"I-{entity_type}"
        if label not in label_to_id:
            raise SchemaValidationError("分词器对齐使用了未知BIO标签。")
        label_ids.append(label_to_id[label])
        loss_mask.append(True)
    return AlignmentResult(tuple(label_ids), tuple(loss_mask), tuple(issues))
