"""训练分区的流式读取、完整性校验和可重放洗牌。"""

from __future__ import annotations

import hashlib
import json
import random
from itertools import batched
from pathlib import Path
from typing import Any, Iterable, Iterator

from anitopy_ml.errors import InputValidationError, SchemaValidationError
from anitopy_ml.training.augment import augment_train_records


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """逐行读取并报告损坏记录的位置，不缓存完整文件。"""
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError
                except (ValueError, TypeError) as error:
                    raise InputValidationError(f"分区JSON无效：{path}:{line_number}。") from error
                yield record
    except (OSError, UnicodeError) as error:
        raise InputValidationError(f"无法读取训练分区：{path}。") from error


def scan_partition(path: Path, *, augment_whitespace: bool = False) -> dict[str, Any]:
    """低内存统计样本数和哈希，恢复时检测实际文件变更。"""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise InputValidationError(f"无法读取训练分区：{path}。") from error
    count = augmented = 0
    for record in iter_jsonl(path):
        text, characters, labels = record.get("text"), record.get("characters"), record.get("labels")
        if not isinstance(text, str) or not text or characters != list(text) or not isinstance(labels, list) or len(labels) != len(text):
            raise SchemaValidationError(f"训练记录的原文、字符与BIO标签不一致：{path}。")
        count += 1
        if augment_whitespace:
            augmented += len(augment_train_records([record]))
    if not count:
        raise InputValidationError(f"训练分区为空：{path}。")
    return {"样本数": count, "增强样本数": augmented, "SHA256": digest.hexdigest()}


def shuffled_records(records: Iterable[dict[str, Any]], *, buffer_size: int, seed: int) -> Iterator[dict[str, Any]]:
    """以固定大小的块洗牌，独立随机源确保恢复后顺序一致。"""
    if buffer_size <= 0:
        raise InputValidationError("洗牌缓冲区大小必须大于0。")
    random_source = random.Random(seed)
    for group in batched(records, buffer_size):
        buffer = list(group)
        random_source.shuffle(buffer)
        yield from buffer


def training_records(path: Path, *, augment_whitespace: bool, buffer_size: int, seed: int) -> Iterator[dict[str, Any]]:
    """按需追加训练增强；验证和冻结测试不经过此入口。"""
    def expanded() -> Iterator[dict[str, Any]]:
        for record in iter_jsonl(path):
            yield record
            if augment_whitespace:
                yield from augment_train_records([record])

    yield from shuffled_records(expanded(), buffer_size=buffer_size, seed=seed)
