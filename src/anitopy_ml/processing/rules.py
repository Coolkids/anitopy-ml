"""把有序名称处理规则编译为作用于当前文本的编辑区间。"""

from __future__ import annotations

from collections.abc import Sequence

from anitopy_ml.errors import PreprocessingError
from anitopy_ml.processing.types import ProcessingRule, ReplacementToken, TextEdit


def _expand(rule: ProcessingRule, match: object) -> str:
    """按受限模板展开正则捕获组。"""
    if rule.type.endswith("mask"):
        return ""
    if rule.type == "string_replace":
        return rule.replacement or ""
    groups = match.groupdict()  # type: ignore[attr-defined]
    result: list[str] = []
    for token in rule.replacement_tokens:
        if token.literal is not None:
            result.append(token.literal)
        elif token.group is not None:
            try:
                value = groups.get(token.group, "") if isinstance(token.group, str) else match.group(token.group)  # type: ignore[attr-defined]
            except (IndexError, KeyError) as error:
                raise PreprocessingError(
                    f"规则{rule.id}的捕获组无法展开。",
                    code="PREPROCESSING_ZERO_WIDTH",
                ) from error
            result.append(value or "")
    return "".join(result)


def edits_for_rule(rule: ProcessingRule, text: str) -> tuple[TextEdit, ...]:
    """返回从左到右、不重叠且坐标属于当前步骤输入的编辑。"""
    if not rule.enabled:
        return ()
    edits: list[TextEdit] = []
    if rule.type.startswith("string_"):
        assert rule.match is not None
        cursor = 0
        while cursor <= len(text) - len(rule.match):
            if rule.count and len(edits) >= rule.count:
                break
            start = text.find(rule.match, cursor)
            if start < 0:
                break
            end = start + len(rule.match)
            replacement = _expand(rule, None)  # type: ignore[arg-type]
            edits.append(TextEdit(start, end, replacement))
            cursor = end
        return tuple(edits)

    if rule.pattern is None:
        return ()
    for match in rule.pattern.finditer(text):
        if rule.count and len(edits) >= rule.count:
            break
        if match.start() >= match.end():
            raise PreprocessingError(
                f"规则{rule.id}在位置{match.start()}匹配了零宽文本。",
                code="PREPROCESSING_ZERO_WIDTH",
            )
        edits.append(TextEdit(match.start(), match.end(), _expand(rule, match)))
    return tuple(edits)


def apply_text_edits(text: str, edits: Sequence[TextEdit]) -> str:
    """将一组有序非重叠编辑一次性作用于原文本。"""
    output: list[str] = []
    previous_end = 0
    for edit in edits:
        if not previous_end <= edit.start < edit.end <= len(text):
            raise ValueError("规则编辑范围无效或发生重叠。")
        output.append(text[previous_end : edit.start])
        output.append(edit.replacement)
        previous_end = edit.end
    output.append(text[previous_end:])
    return "".join(output)
