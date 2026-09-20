"""标题规范化、噪声识别及原文位置映射。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from anitopy_ml.errors import InputValidationError, SchemaValidationError


@dataclass(frozen=True, slots=True)
class SourceRange:
    """原始标题中的左闭右开字符区间。"""

    start: int
    end: int

    def validate(self, raw_text: str) -> None:
        """验证区间位于原始标题中。"""
        if not 0 <= self.start <= self.end <= len(raw_text):
            raise SchemaValidationError("原始文本区间超出标题边界。")


@dataclass(frozen=True, slots=True)
class CharacterMapping:
    """规范化后一个字符所对应的原文区间。"""

    source: SourceRange


@dataclass(frozen=True, slots=True)
class NoiseRegion:
    """被模型视图隐藏、但仍可追溯的原文噪声区域。"""

    source: SourceRange
    kind: str


@dataclass(frozen=True, slots=True)
class TextWindow:
    """模型文本的一段窗口及其可追溯的全局位置。"""

    text: str
    model_start: int
    model_end: int
    source: SourceRange


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """原文、规范化文本、模型文本和两者之间的映射。"""

    raw_text: str
    normalized_text: str
    model_text: str
    normalized_mapping: tuple[CharacterMapping, ...]
    model_mapping: tuple[CharacterMapping, ...]
    noise_regions: tuple[NoiseRegion, ...]
    version: str = "1.0"

    def __post_init__(self) -> None:
        """确保每个输出字符均有一条原文映射。"""
        if len(self.normalized_text) != len(self.normalized_mapping):
            raise SchemaValidationError("规范化文本与字符映射长度不一致。")
        if len(self.model_text) != len(self.model_mapping):
            raise SchemaValidationError("模型文本与字符映射长度不一致。")
        for mapping in (*self.normalized_mapping, *self.model_mapping):
            mapping.source.validate(self.raw_text)
        for region in self.noise_regions:
            region.source.validate(self.raw_text)

    def source_range_for_normalized(self, start: int, end: int) -> SourceRange:
        """将规范化文本区间映射回覆盖它的最小原文区间。"""
        return _source_range_for_output(self.normalized_mapping, start, end)

    def source_range_for_model(self, start: int, end: int) -> SourceRange:
        """将模型文本区间映射回覆盖它的最小原文区间。"""
        return _source_range_for_output(self.model_mapping, start, end)

    def normalized_ranges_for_source(self, start: int, end: int) -> tuple[SourceRange, ...]:
        """找出与原文区间相交的规范化文本区间，可含多个不连续片段。"""
        return _output_ranges_for_source(self.normalized_mapping, start, end)

    def make_model_windows(self, *, max_chars: int = 4096, stride: int = 256) -> tuple[TextWindow, ...]:
        """按字符生成覆盖全部模型文本的重叠窗口，供后续词元滑窗复用。"""
        if max_chars <= 0:
            raise SchemaValidationError("窗口最大字符数必须大于0。")
        if not 0 <= stride < max_chars:
            raise SchemaValidationError("窗口重叠长度必须大于等于0且小于窗口最大字符数。")
        if not self.model_text:
            return ()

        windows: list[TextWindow] = []
        start = 0
        while start < len(self.model_text):
            end = min(start + max_chars, len(self.model_text))
            windows.append(
                TextWindow(
                    text=self.model_text[start:end],
                    model_start=start,
                    model_end=end,
                    source=self.source_range_for_model(start, end),
                )
            )
            if end == len(self.model_text):
                break
            start = end - stride
        return tuple(windows)


_URL_PATTERN = re.compile(r"https?://[^\s\[\]<>]+", re.IGNORECASE)
_BBCODE_TAG_PATTERN = re.compile(
    r"\[/?(?:img|url|size|b|i|color|font|quote)(?:=[^\]]*)?\]",
    re.IGNORECASE,
)


def _source_range_for_output(
    mappings: tuple[CharacterMapping, ...], start: int, end: int
) -> SourceRange:
    """由输出位置映射计算原文覆盖区间。"""
    if not 0 <= start < end <= len(mappings):
        raise SchemaValidationError("输出文本区间超出边界。")
    selected = mappings[start:end]
    return SourceRange(
        start=min(item.source.start for item in selected),
        end=max(item.source.end for item in selected),
    )


def _output_ranges_for_source(
    mappings: tuple[CharacterMapping, ...], start: int, end: int
) -> tuple[SourceRange, ...]:
    """合并所有与给定原文区间相交的连续输出位置。"""
    if start > end:
        raise SchemaValidationError("原文区间起点不能大于终点。")
    positions = [
        index
        for index, mapping in enumerate(mappings)
        if mapping.source.start < end and start < mapping.source.end
    ]
    if not positions:
        return ()
    ranges: list[SourceRange] = []
    range_start = previous = positions[0]
    for index in positions[1:]:
        if index != previous + 1:
            ranges.append(SourceRange(range_start, previous + 1))
            range_start = index
        previous = index
    ranges.append(SourceRange(range_start, previous + 1))
    return tuple(ranges)


def _collapse_whitespace(
    characters: list[tuple[str, SourceRange]]
) -> list[tuple[str, SourceRange]]:
    """将连续空白合并为一个空格，并合并其原文覆盖范围。"""
    collapsed: list[tuple[str, SourceRange]] = []
    for character, source in characters:
        if character.isspace():
            if collapsed and collapsed[-1][0] == " ":
                previous_source = collapsed[-1][1]
                collapsed[-1] = (" ", SourceRange(previous_source.start, source.end))
            else:
                collapsed.append((" ", source))
        else:
            collapsed.append((character, source))
    return collapsed


def _normalize_characters(raw_text: str) -> list[tuple[str, SourceRange]]:
    """逐字符执行NFKC，保留一对多转换的原文映射。"""
    characters: list[tuple[str, SourceRange]] = []
    for index, character in enumerate(raw_text):
        normalized = unicodedata.normalize("NFKC", character)
        source = SourceRange(index, index + 1)
        for output_character in normalized:
            characters.append((output_character, source))
    return _collapse_whitespace(characters)


def _noise_regions(
    normalized_text: str, mappings: tuple[CharacterMapping, ...]
) -> tuple[NoiseRegion, ...]:
    """从规范化文本中识别网址和常见BBCode标签的原文区间。"""
    candidates: list[tuple[int, int, str]] = []
    candidates.extend((match.start(), match.end(), "url") for match in _URL_PATTERN.finditer(normalized_text))
    candidates.extend(
        (match.start(), match.end(), "bbcode") for match in _BBCODE_TAG_PATTERN.finditer(normalized_text)
    )
    regions: list[NoiseRegion] = []
    for start, end, kind in sorted(candidates):
        source = _source_range_for_output(mappings, start, end)
        if regions and source.start <= regions[-1].source.end:
            previous = regions[-1]
            regions[-1] = NoiseRegion(
                source=SourceRange(previous.source.start, max(previous.source.end, source.end)),
                kind="mixed" if previous.kind != kind else kind,
            )
        else:
            regions.append(NoiseRegion(source=source, kind=kind))
    return tuple(regions)


def normalize_title(raw_text: str) -> NormalizationResult:
    """生成可追溯的规范化视图和移除网页噪声后的模型视图。"""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise InputValidationError("标题不能为空。")

    normalized_characters = _normalize_characters(raw_text)
    normalized_text = "".join(item[0] for item in normalized_characters)
    normalized_mapping = tuple(CharacterMapping(item[1]) for item in normalized_characters)
    noise_regions = _noise_regions(normalized_text, normalized_mapping)

    model_characters: list[tuple[str, SourceRange]] = []
    noise_index = 0
    for character, source in normalized_characters:
        while noise_index < len(noise_regions) and source.start >= noise_regions[noise_index].source.end:
            noise_index += 1
        hidden = (
            noise_index < len(noise_regions)
            and source.start < noise_regions[noise_index].source.end
            and noise_regions[noise_index].source.start < source.end
        )
        if not hidden:
            model_characters.append((character, source))

    model_characters = _collapse_whitespace(model_characters)
    model_text = "".join(item[0] for item in model_characters).strip()
    left_trim = len("".join(item[0] for item in model_characters)) - len(
        "".join(item[0] for item in model_characters).lstrip()
    )
    right_trimmed = model_characters[left_trim:]
    while right_trimmed and right_trimmed[-1][0] == " ":
        right_trimmed.pop()
    return NormalizationResult(
        raw_text=raw_text,
        normalized_text=normalized_text,
        model_text=model_text,
        normalized_mapping=normalized_mapping,
        model_mapping=tuple(CharacterMapping(item[1]) for item in right_trimmed),
        noise_regions=noise_regions,
    )
