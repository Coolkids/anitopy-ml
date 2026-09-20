"""仅对训练分区执行且可追溯的保守文本增强。"""

from __future__ import annotations

from typing import Any, Iterable

from anitopy_ml.errors import SchemaValidationError


def augment_train_whitespace(record: dict[str, Any]) -> dict[str, Any] | None:
    """在首个空白后插入一个空白，并同步重建字符与BIO标签。"""
    if record.get("split") != "train":
        raise SchemaValidationError("数据增强只能作用于训练分区。")
    text = str(record["text"])
    characters = list(record["characters"])
    labels = list(record["labels"])
    if text != "".join(characters) or len(characters) != len(labels):
        raise SchemaValidationError("增强输入的文本、字符和标签不一致。")
    try:
        position = next(index for index, character in enumerate(characters) if character.isspace())
    except StopIteration:
        return None
    augmented_characters = [*characters[: position + 1], " ", *characters[position + 1 :]]
    augmented_labels = [*labels[: position + 1], "O", *labels[position + 1 :]]
    return {
        **record,
        "sample_id": f"{record['sample_id']}:aug-space",
        "parent_sample_id": record["sample_id"],
        "augmentation": "在既有空白后插入空白",
        "text": "".join(augmented_characters),
        "characters": augmented_characters,
        "labels": augmented_labels,
    }


def augment_train_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """仅生成可安全增强的训练样本，保留原分区和作品组追溯字段。"""
    augmented: list[dict[str, Any]] = []
    for record in records:
        if record.get("split") != "train":
            continue
        item = augment_train_whitespace(record)
        if item is not None:
            augmented.append(item)
    return augmented
