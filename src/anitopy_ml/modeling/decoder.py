"""受BIO转移约束的Viterbi解码。"""

from __future__ import annotations

from math import inf
from typing import Sequence

from anitopy_ml.errors import SchemaValidationError


def _parts(label: str) -> tuple[str, str | None]:
    """拆分BIO标签。"""
    if label == "O":
        return "O", None
    prefix, separator, entity_type = label.partition("-")
    if separator != "-" or prefix not in {"B", "I"} or not entity_type:
        raise SchemaValidationError("BIO标签格式无效。")
    return prefix, entity_type


def transition_allowed(previous: str | None, current: str) -> bool:
    """判断当前BIO标签是否可接在前一个标签后。"""
    prefix, entity_type = _parts(current)
    if previous is None:
        return prefix != "I"
    previous_prefix, previous_type = _parts(previous)
    if prefix != "I":
        return True
    return previous_prefix in {"B", "I"} and previous_type == entity_type


def constrained_bio_decode(
    logits: Sequence[Sequence[float]],
    labels: Sequence[str],
) -> list[str]:
    """从每位置类别分数中寻找分数最高且BIO合法的标签路径。"""
    if not logits or not labels:
        raise SchemaValidationError("解码分数和标签不能为空。")
    if any(len(row) != len(labels) for row in logits):
        raise SchemaValidationError("每个位置的类别分数必须与标签数一致。")
    for label in labels:
        _parts(label)
    scores = [-inf] * len(labels)
    paths: list[list[int]] = []
    for index, label in enumerate(labels):
        if transition_allowed(None, label):
            scores[index] = float(logits[0][index])
    paths.append([-1] * len(labels))
    for row in logits[1:]:
        next_scores = [-inf] * len(labels)
        backpointers = [-1] * len(labels)
        for current_index, current_label in enumerate(labels):
            best_score = -inf
            best_previous = -1
            for previous_index, previous_label in enumerate(labels):
                if scores[previous_index] == -inf or not transition_allowed(previous_label, current_label):
                    continue
                candidate = scores[previous_index] + float(row[current_index])
                if candidate > best_score:
                    best_score, best_previous = candidate, previous_index
            if best_previous >= 0:
                next_scores[current_index] = best_score
                backpointers[current_index] = best_previous
        scores = next_scores
        paths.append(backpointers)
    last = max(range(len(labels)), key=lambda index: scores[index])
    if scores[last] == -inf:
        raise SchemaValidationError("没有满足BIO约束的可用解码路径。")
    indices = [last]
    for position in range(len(paths) - 1, 0, -1):
        indices.append(paths[position][indices[-1]])
    indices.reverse()
    return [labels[index] for index in indices]
