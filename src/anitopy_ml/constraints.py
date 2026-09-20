"""媒体标题中明确格式字段的提取、规范化和约束校验。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from anitopy_ml.errors import SchemaValidationError
from anitopy_ml.normalizer import SourceRange, normalize_title
from anitopy_ml.schemas import ExtractedFields


@dataclass(frozen=True, slots=True)
class FieldHint:
    """由确定性格式识别出的原文片段和规范化值。"""

    label: str
    source: SourceRange
    raw: str
    value: str


@dataclass(slots=True)
class ConstraintResult:
    """明确格式提取出的字段、证据提示和非阻断告警。"""

    fields: ExtractedFields = field(default_factory=ExtractedFields)
    hints: list[FieldHint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_SEASON_EPISODE = re.compile(
    r"(?i)(?P<season_expr>S(?P<season>\d{1,2}))(?P<episode_expr>E(?P<episode>\d+(?:\.\d+)?))"
)
_SEASON_ONLY = re.compile(r"(?i)\bS(?P<season>\d{1,2})(?![A-Z0-9])")
_CHINESE_SEASON = re.compile(r"第?\s*(?P<season>[0-9一二三四五六七八九十百]+)\s*季")
_RANGE = re.compile(r"(?i)(?:EP|E)?\s*(?P<start>\d+(?:\.\d+)?)\s*(?:~|-|至|到)\s*(?:EP|E)?\s*(?P<end>\d+(?:\.\d+)?)")
_EPISODE = re.compile(r"(?i)\b(?:EP|E|Episode)\s*(?P<episode>\d+(?:\.\d+)?)")
_CHINESE_EPISODE = re.compile(r"第\s*(?P<episode>[0-9一二三四五六七八九十百]+)\s*[集话話]")
_DECLARED_COUNT = re.compile(r"全\s*(?P<count>\d+)\s*[集话話]")
_EPISODE_LIST = re.compile(r"(?<![A-Za-z0-9])(?P<first>\d+(?:\.\d+)?)\s*&\s*(?P<second>\d+(?:\.\d+)?)")
_SPECIAL_TYPES = re.compile(r"(?i)\b(?P<kind>OVA|OAD|TVSP|NCOP|NCED|SP)\b")

_SOURCE_TERMS = ("BDRemux", "BluRay", "WEB-DL", "WEBRip", "HDTV", "HDTS", "DVD")
_VIDEO_CODECS = {"H.264": "H.264", "H264": "H.264", "AVC": "H.264", "HEVC": "H.265", "H.265": "H.265", "H265": "H.265"}
_VIDEO_ENCODERS = ("x264", "x265")
_AUDIO_CODECS = ("AAC", "FLAC", "AC3", "EAC3", "DDP", "DTS")
_RESOLUTION = re.compile(r"(?i)\b(?:480p|720p|1080p|1080i|2160p|4K|8K)\b")
_SUBTITLE_MODE = {"内封": "内封", "内嵌": "内嵌", "外挂": "外挂"}
_SUBTITLE_LANGUAGE = {"简体中文": "zh-Hans", "繁体中文": "zh-Hant", "简繁": "zh-Hans,zh-Hant", "简/繁": "zh-Hans,zh-Hant"}


def _number(value: str) -> str | None:
    """将正数转换为无前导零、无浮点误差的十进制字符串。"""
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    if number <= 0:
        return None
    return format(number.normalize(), "f")


def _chinese_number(value: str) -> int | None:
    """解析第一版需要的常见中文季集数字。"""
    if value.isdecimal():
        return int(value)
    values = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100}
    if any(character not in values for character in value):
        return None
    if value == "十":
        return 10
    if "十" in value:
        before, _, after = value.partition("十")
        tens = values[before] if before else 1
        ones = values[after] if after else 0
        return tens * 10 + ones
    return values.get(value)


def _append_hint(result: ConstraintResult, label: str, start: int, end: int, raw_text: str, value: str, normalized) -> None:
    """将模型文本位置转换为原文证据后保存。"""
    source = normalized.source_range_for_model(start, end)
    result.hints.append(FieldHint(label, source, raw_text[source.start:source.end], value))


def _unique_append(values: list[str], value: str) -> None:
    """按出现顺序写入不重复的字符串字段。"""
    if value not in values:
        values.append(value)


def extract_constraints(raw_text: str) -> ConstraintResult:
    """从明确格式提取字段；不以孤立数字猜测季数或集数。"""
    normalized = normalize_title(raw_text)
    text = normalized.model_text
    result = ConstraintResult()
    occupied: list[tuple[int, int]] = []

    for match in _SEASON_EPISODE.finditer(text):
        season = int(match.group("season"))
        episode = _number(match.group("episode"))
        if season > 0 and episode:
            if season not in result.fields.seasons:
                result.fields.seasons.append(season)
            result.fields.episodes.append({"raw": match.group("episode"), "value": episode, "numbering": "season"})
            _append_hint(result, "SEASON_EXPR", match.start("season_expr"), match.end("season_expr"), raw_text, str(season), normalized)
            _append_hint(result, "EPISODE_EXPR", match.start("episode_expr"), match.end("episode_expr"), raw_text, episode, normalized)
            occupied.append((match.start(), match.end()))

    for match in _RANGE.finditer(text):
        start = _number(match.group("start"))
        end = _number(match.group("end"))
        if not start or not end or Decimal(start) > Decimal(end):
            result.warnings.append("已忽略起止顺序无效的集数范围。")
            continue
        result.fields.episode_ranges.append({"raw": match.group(0), "start": start, "end": end, "numbering": "unknown"})
        _append_hint(result, "EPISODE_EXPR", match.start(), match.end(), raw_text, f"{start}-{end}", normalized)
        occupied.append((match.start(), match.end()))

    for match in _EPISODE_LIST.finditer(text):
        for name in ("first", "second"):
            value = _number(match.group(name))
            if value:
                result.fields.episodes.append({"raw": match.group(name), "value": value, "numbering": "unknown"})
        _append_hint(result, "EPISODE_EXPR", match.start(), match.end(), raw_text, "列表", normalized)
        occupied.append((match.start(), match.end()))

    for match in _EPISODE.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in occupied):
            continue
        value = _number(match.group("episode"))
        if value:
            result.fields.episodes.append({"raw": match.group("episode"), "value": value, "numbering": "unknown"})
            _append_hint(result, "EPISODE_EXPR", match.start(), match.end(), raw_text, value, normalized)

    for match in _CHINESE_SEASON.finditer(text):
        value = _chinese_number(match.group("season"))
        if value and value > 0 and value not in result.fields.seasons:
            result.fields.seasons.append(value)
            _append_hint(result, "SEASON_EXPR", match.start(), match.end(), raw_text, str(value), normalized)
    for match in _SEASON_ONLY.finditer(text):
        value = int(match.group("season"))
        if value > 0 and value not in result.fields.seasons:
            result.fields.seasons.append(value)
            _append_hint(result, "SEASON_EXPR", match.start(), match.end(), raw_text, str(value), normalized)
    for match in _CHINESE_EPISODE.finditer(text):
        value = _chinese_number(match.group("episode"))
        if value and value > 0:
            result.fields.episodes.append({"raw": match.group("episode"), "value": str(value), "numbering": "unknown"})
            _append_hint(result, "EPISODE_EXPR", match.start(), match.end(), raw_text, str(value), normalized)
    for match in _DECLARED_COUNT.finditer(text):
        count = int(match.group("count"))
        if count > 0:
            result.fields.declared_episode_count = count
            _append_hint(result, "EPISODE_COUNT_EXPR", match.start(), match.end(), raw_text, str(count), normalized)
    special = _SPECIAL_TYPES.search(text)
    if special:
        result.fields.special_type = special.group("kind").upper()
        _append_hint(result, "SPECIAL_TYPE", special.start(), special.end(), raw_text, result.fields.special_type, normalized)

    for term in _SOURCE_TERMS:
        if re.search(re.escape(term), text, re.IGNORECASE):
            _unique_append(result.fields.source, term)
    for raw, canonical in _VIDEO_CODECS.items():
        if re.search(re.escape(raw), text, re.IGNORECASE):
            _unique_append(result.fields.video_codecs, canonical)
    for term in _VIDEO_ENCODERS:
        if re.search(re.escape(term), text, re.IGNORECASE):
            _unique_append(result.fields.video_encoders, term)
    for term in _AUDIO_CODECS:
        if re.search(rf"(?i)(?<![A-Z0-9]){re.escape(term)}(?=\b|\d)", text):
            _unique_append(result.fields.audio_codecs, term)
    for match in _RESOLUTION.finditer(text):
        _unique_append(result.fields.resolution, match.group(0))
    for raw, canonical in _SUBTITLE_MODE.items():
        if raw in text:
            result.fields.subtitle_mode = canonical
    for raw, canonical in _SUBTITLE_LANGUAGE.items():
        if raw in text:
            for language in canonical.split(","):
                _unique_append(result.fields.subtitle_languages, language)

    if result.fields.special_type:
        result.fields.release_kind = "special"
    elif result.fields.declared_episode_count is not None:
        result.fields.release_kind = "season_pack"
    elif result.fields.episodes or result.fields.episode_ranges:
        result.fields.release_kind = "episode"
    validate_extracted_fields(result.fields)
    return result


def validate_extracted_fields(fields: ExtractedFields) -> None:
    """校验结构化季集字段，防止范围、类型与数值自相矛盾。"""
    if any(not isinstance(item, int) or item <= 0 for item in fields.seasons):
        raise SchemaValidationError("季数必须为正整数。")
    if len(fields.seasons) != len(set(fields.seasons)):
        raise SchemaValidationError("季数不能重复。")
    if fields.declared_episode_count is not None and fields.declared_episode_count <= 0:
        raise SchemaValidationError("声明总集数必须为正整数。")
    for item in fields.episodes:
        if set(item) != {"raw", "value", "numbering"} or _number(item["value"]) is None:
            raise SchemaValidationError("集数字段结构或数值无效。")
    for item in fields.episode_ranges:
        required = {"raw", "start", "end", "numbering"}
        if set(item) != required or not _number(item["start"]) or not _number(item["end"]):
            raise SchemaValidationError("集数范围字段结构或数值无效。")
        if Decimal(item["start"]) > Decimal(item["end"]):
            raise SchemaValidationError("集数范围起点不能大于终点。")
