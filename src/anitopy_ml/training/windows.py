"""长标题的字符窗口与重叠损失掩码。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from anitopy_ml.errors import SchemaValidationError


@dataclass(frozen=True, slots=True)
class CharacterWindow:
    """带全局偏移和去重损失掩码的一段训练文本。"""

    text: str
    labels: tuple[str, ...]
    global_start: int
    global_end: int
    loss_mask: tuple[bool, ...]


def make_character_windows(
    text: str,
    labels: Sequence[str],
    *,
    max_chars: int,
    stride: int,
) -> tuple[CharacterWindow, ...]:
    """切分长标题；重叠字符只在首次出现的窗口计算损失。"""
    if len(text) != len(labels):
        raise SchemaValidationError("窗口文本和标签长度不一致。")
    if max_chars <= 0 or not 0 <= stride < max_chars:
        raise SchemaValidationError("窗口参数无效。")
    windows: list[CharacterWindow] = []
    seen: set[int] = set()
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        positions = range(start, end)
        mask = tuple(position not in seen for position in positions)
        seen.update(positions)
        windows.append(
            CharacterWindow(text[start:end], tuple(labels[start:end]), start, end, mask)
        )
        if end == len(text):
            break
        start = end - stride
    return tuple(windows)
