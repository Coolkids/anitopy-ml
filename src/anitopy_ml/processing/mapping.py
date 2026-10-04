"""执行名称编辑并合成处理后文本到原始标题的位置映射。"""

from __future__ import annotations

from collections.abc import Sequence

from anitopy_ml.errors import SchemaValidationError
from anitopy_ml.normalizer import SourceRange
from anitopy_ml.processing.types import TextEdit, TextState


def _merge_sources(ranges: Sequence[SourceRange]) -> tuple[SourceRange, ...]:
    """合并重复、重叠或相邻来源区间，但保留删除造成的间隙。"""
    if not ranges:
        return ()
    ordered = sorted(set(ranges), key=lambda item: (item.start, item.end))
    merged: list[SourceRange] = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.start <= previous.end:
            merged[-1] = SourceRange(previous.start, max(previous.end, current.end))
        else:
            merged.append(current)
    return tuple(merged)


def apply_edits(state: TextState, edits: Sequence[TextEdit]) -> TextState:
    """将单步有序编辑应用到文本和字符来源映射。"""
    output_text: list[str] = []
    output_mappings: list[tuple[SourceRange, ...]] = []
    previous_end = 0
    for edit in edits:
        if not previous_end <= edit.start < edit.end <= len(state.text):
            raise SchemaValidationError("预处理编辑范围无效或发生重叠。")
        output_text.append(state.text[previous_end : edit.start])
        output_mappings.extend(state.mappings[previous_end : edit.start])
        origins = _merge_sources(
            [origin for mapping in state.mappings[edit.start : edit.end] for origin in mapping]
        )
        output_text.append(edit.replacement)
        if state.text[edit.start : edit.end] == edit.replacement:
            output_mappings.extend(state.mappings[edit.start : edit.end])
        else:
            output_mappings.extend(origins for _ in edit.replacement)
        previous_end = edit.end
    output_text.append(state.text[previous_end:])
    output_mappings.extend(state.mappings[previous_end:])
    text = "".join(output_text)
    if len(text) != len(output_mappings):
        raise SchemaValidationError("预处理文本与原文位置映射长度不一致。")
    return TextState(state.raw_text, text, tuple(output_mappings))
