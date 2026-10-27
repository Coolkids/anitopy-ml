"""按用户维护的标题模板生成可追溯的合成 BIO 训练数据。"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from anitopy_ml.errors import InputValidationError, SchemaValidationError
from anitopy_ml.schemas import BIO_LABELS


_PLACEHOLDER = re.compile(r"(?P<marker>[$&])(?P<field>[a-z_][a-z0-9_]*)(?:\[(?P<index>\d+)\])?")
_RANDOM_INTEGER = re.compile(r"%random\((?P<width>\d+)\)")
_RANDOM_RANGE = re.compile(r"%random_range\((?P<start>-?\d+)-(?P<end>-?\d+)\)")
_RANDOM_STRING = re.compile(
    r"%random_str\(\s*(?P<source>\$[a-z_][a-z0-9_]*(?:\[\d+\])?)\s*,\s*(?P<maximum>\d+)\s*\)"
)
_RANDOM_INTEGER_MARKER = re.compile(r"%random")
_DISABLE_LENGTH_CONDITION = re.compile(
    r"^\s*%len\(\s*\$(?P<field>[a-z_][a-z0-9_]*)\s*\)\s*"
    r"(?P<operator>>=|<=|==|!=|>|<)\s*(?P<count>\d+)\s*$"
)
_FIELD_LABELS = {
    "title": "TITLE",
    "title_alias": "TITLE_ALIAS",
    "season_expr": "SEASON_EXPR",
    "episode_expr": "EPISODE_EXPR",
    "episode_count_expr": "EPISODE_COUNT_EXPR",
    "release_group": "RELEASE_GROUP",
    "source": "SOURCE",
    "resolution": "RESOLUTION",
    "video_term": "VIDEO_TERM",
    "audio_term": "AUDIO_TERM",
    "subtitle_language_chs": "SUBTITLE_LANGUAGE",
    "subtitle_language_en": "SUBTITLE_LANGUAGE",
    "subtitle_mode": "SUBTITLE_MODE",
    "file_extension": "FILE_EXTENSION",
    "bit_depth": "BIT_DEPTH",
}

_TEMPLATE_LOCAL_FIELD_LABELS = {
    "noise": "NOISE",
    "noise2": "NOISE",
    "release_version": "RELEASE_VERSION",
}

_TITLE_FIELD_LABELS = {
    "title": "TITLE",
    "title_en": "TITLE",
    "title_cn": "TITLE_ALIAS",
    "title_as_title_alias_1": "TITLE_ALIAS",
}

_TITLE_TEMPLATE_FIELDS = frozenset({*_TITLE_FIELD_LABELS, "title_alias"})


@dataclass(frozen=True, slots=True)
class RenderedValue:
    """占位符替换文本及其中相对位置的实体标签。"""

    text: str
    spans: tuple[tuple[int, int, str], ...]


def load_template_data(path: str | Path) -> dict[str, Any]:
    """读取并校验模板数据的最低结构。"""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InputValidationError("无法读取模板数据JSON。") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("title"), list):
        raise SchemaValidationError("模板数据缺少title列表。")
    if not isinstance(payload.get("name_template"), list):
        raise SchemaValidationError("模板数据缺少name_template列表。")
    for field in _FIELD_LABELS:
        if field in {"title", "title_alias", "season_expr", "release_version"}:
            continue
        if not isinstance(payload.get(field), list) or not payload[field]:
            raise SchemaValidationError(f"模板数据字段{field}必须是非空列表。")
    return payload


def _season_ranges(text: str, season_values: Iterable[str]) -> list[tuple[int, int]]:
    """找出标题中明确声明的季数表达，优先匹配较长表达。"""
    ranges: list[tuple[int, int]] = []
    for value in sorted({item for item in season_values if item}, key=len, reverse=True):
        start = text.find(value)
        while start >= 0:
            end = start + len(value)
            if not any(start < other_end and end > other_start for other_start, other_end in ranges):
                ranges.append((start, end))
            start = text.find(value, start + 1)
    return sorted(ranges)


def _title_value(text: str, label: str, season_values: Iterable[str]) -> RenderedValue:
    """把标题中的已知季数表达从标题实体中拆出，保持 BIO 片段不重叠。"""
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for start, end in _season_ranges(text, season_values):
        if cursor < start:
            spans.append((cursor, start, label))
        spans.append((start, end, "SEASON_EXPR"))
        cursor = end
    if cursor < len(text):
        spans.append((cursor, len(text), label))
    return RenderedValue(text, tuple(spans))


def _title_text(title_item: dict[str, Any], field: str) -> str:
    """读取标题条目中的单个名称字段，并拒绝非文本配置。"""
    value = title_item.get(field, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise SchemaValidationError(f"标题条目的{field}必须是文本。")
    return value


def _aliases(title_item: dict[str, Any]) -> list[str]:
    """读取旧版别名列表，过滤空值以兼容既有数据文件。"""
    values = title_item.get("title_alias", [])
    if not isinstance(values, list):
        raise SchemaValidationError("标题条目的title_alias必须是列表。")
    return [str(item) for item in values if str(item)]


def _english_title(title_item: dict[str, Any], aliases: list[str]) -> str:
    """优先读取显式英文主标题，兼容旧版第一个别名。"""
    explicit_title = _title_text(title_item, "title_en")
    return explicit_title if explicit_title.strip() else (aliases[0] if aliases else "")


def _chinese_title(title_item: dict[str, Any]) -> str:
    """优先读取显式中文名称，兼容旧版主标题字段。"""
    explicit_title = _title_text(title_item, "title_cn")
    return explicit_title if explicit_title.strip() else _title_text(title_item, "title")


def _template_title_value(
    title_item: dict[str, Any], field: str, aliases: list[str]
) -> str:
    """根据模板标题占位符返回其对应的名称文本。"""
    if field == "title_en":
        return _english_title(title_item, aliases)
    if field == "title_cn":
        return _chinese_title(title_item)
    if field == "title_as_title_alias_1":
        return _title_text(title_item, "title")
    return _title_text(title_item, field)


def _work_key(title_item: dict[str, Any]) -> str:
    """优先使用英文主标题构造分区作品键，兼容旧版标题字段。"""
    aliases = _aliases(title_item)
    for value in (_english_title(title_item, aliases), _title_text(title_item, "title"), _chinese_title(title_item)):
        value = value.strip()
        if value:
            return value.casefold()
    raise SchemaValidationError("标题条目至少需要title_en、title或title_cn中的一个非空文本。")


def _title_alias_value(
    aliases: Iterable[str],
    *,
    open_text: str,
    begin_text: str,
    close_text: str,
    separator: str,
    season_values: Iterable[str],
) -> RenderedValue:
    """逐个包裹并拼接别名，只标记别名本身，不标记包围文本和分隔符。"""
    values = [item for item in aliases if item]
    if not values:
        return RenderedValue("", ())
    parts: list[str] = []
    spans: list[tuple[int, int, str]] = []
    offset = 0
    for index, alias in enumerate(values):
        if index:
            parts.append(separator)
            offset += len(separator)
        else:
            parts.append(begin_text)
            offset += len(begin_text)
        parts.append(open_text)
        offset += len(open_text)
        rendered = _title_value(alias, "TITLE_ALIAS", season_values)
        parts.append(rendered.text)
        spans.extend((offset + start, offset + end, label) for start, end, label in rendered.spans)
        offset += len(alias)
        parts.append(close_text)
        offset += len(close_text)
    return RenderedValue("".join(parts), tuple(spans))


def _is_title_template_field(field: str) -> bool:
    """判断占位符是否会渲染为主标题或别名实体。"""
    return field in _TITLE_TEMPLATE_FIELDS


def _alias_wrapper_boundary(field: str, template_group: dict[str, Any], *, prefix: bool) -> str:
    """返回别名占位符在相邻标题实体一侧提供的可见边界文本。"""
    if field != "title_alias":
        return ""
    if prefix:
        return str(template_group.get("title_alias_begin", "")) + str(
            template_group.get("title_alias_open", "")
        )
    return str(template_group.get("title_alias_close", ""))


def _validate_title_boundaries(
    templates: Iterable[object], title_items: Iterable[dict[str, Any]]
) -> None:
    """拒绝相邻标题实体无边界及多个别名无分隔符的模板。"""
    has_multiple_aliases = any(len(_aliases(item)) > 1 for item in title_items)
    for template_group in templates:
        if not isinstance(template_group, dict):
            raise SchemaValidationError("name_template中的条目必须是对象。")
        names = template_group.get("name")
        if not isinstance(names, list) or not names:
            raise SchemaValidationError("name_template中的name必须是非空列表。")
        for raw_template in names:
            template = str(raw_template)
            placeholders = list(_PLACEHOLDER.finditer(template))
            has_unindexed_alias = any(
                match.group("field") == "title_alias" and match.group("index") is None
                for match in placeholders
            )
            has_alias_wrapper = bool(str(template_group.get("title_alias_open", ""))) or bool(
                str(template_group.get("title_alias_close", ""))
            )
            if (
                has_multiple_aliases
                and has_unindexed_alias
                and not str(template_group.get("title_alias_separator", ""))
                and not has_alias_wrapper
            ):
                raise SchemaValidationError(
                    "模板循环多个title_alias时必须设置非空title_alias_separator，"
                    "或为每个别名设置title_alias_open/title_alias_close。"
                )
            for previous, current in zip(placeholders, placeholders[1:]):
                previous_field = previous.group("field")
                current_field = current.group("field")
                if (
                    previous.end() != current.start()
                    or not _is_title_template_field(previous_field)
                    or not _is_title_template_field(current_field)
                ):
                    continue
                left_boundary = _alias_wrapper_boundary(
                    previous_field, template_group, prefix=False
                )
                right_boundary = _alias_wrapper_boundary(
                    current_field, template_group, prefix=True
                )
                if not left_boundary and not right_boundary:
                    raise SchemaValidationError(
                        "模板中相邻的标题或别名缺少明确分隔符；请在模板中加入空格、/等字符，"
                        "或设置非空title_alias_open/title_alias_begin/title_alias_close。"
                    )


def _template_is_disabled(template_group: dict[str, Any], title_item: dict[str, Any]) -> bool:
    """按标题条目判断模板组是否应被禁用。"""
    conditions = template_group.get("disable", [])
    if conditions is None:
        return False
    if not isinstance(conditions, list) or not all(isinstance(item, str) for item in conditions):
        raise SchemaValidationError("模板disable必须是条件字符串列表。")
    for condition in conditions:
        match = _DISABLE_LENGTH_CONDITION.fullmatch(condition)
        if match is None:
            raise SchemaValidationError(
                "模板disable仅支持%len($数组字段)与整数的>,>=,<,<=,==,!=比较。"
            )
        field = match.group("field")
        value = title_item.get(field)
        if not isinstance(value, list):
            raise SchemaValidationError(f"%len引用的{field}必须是标题条目中的数组字段。")
        actual = len(value)
        expected = int(match.group("count"))
        operator = match.group("operator")
        matched = {
            ">": actual > expected,
            ">=": actual >= expected,
            "<": actual < expected,
            "<=": actual <= expected,
            "==": actual == expected,
            "!=": actual != expected,
        }[operator]
        if matched:
            return True
    return False


def _ordinary_value(value: str, label: str) -> RenderedValue:
    """构造普通字段的一段完整实体。"""
    return RenderedValue(value, ((0, len(value), label),))


def _render_random_integers(value: str, random_source: random.Random) -> str:
    """展开随机数字函数，支持补零位数和不补零范围。"""

    def replace_range(match: re.Match[str]) -> str:
        start, end = int(match.group("start")), int(match.group("end"))
        if start > end:
            raise SchemaValidationError("%random_range 的起点不能大于终点。")
        return str(random_source.randint(start, end))

    def replace(match: re.Match[str]) -> str:
        width = int(match.group("width"))
        if not 1 <= width <= 9:
            raise SchemaValidationError("%random 的位数必须在1到9之间。")
        return f"{random_source.randint(1, 10**width - 1):0{width}d}"

    rendered = _RANDOM_RANGE.sub(replace_range, value)
    rendered = _RANDOM_INTEGER.sub(replace, rendered)
    if _RANDOM_INTEGER_MARKER.search(rendered):
        raise SchemaValidationError("随机数字占位符必须写为%random(位数)或%random_range(起点-终点)。")
    return rendered


def _render_random_strings(
    value: str, values: dict[str, RenderedValue], random_source: random.Random
) -> str:
    """从指定字段中随机截取不超过上限的连续文本。"""

    def replace(match: re.Match[str]) -> str:
        source_key = match.group("source")[1:]
        source = values.get(source_key)
        if source is None:
            raise SchemaValidationError(f"%random_str引用了尚未定义的字符串来源：{source_key}。")
        maximum = int(match.group("maximum"))
        if maximum <= 0:
            raise SchemaValidationError("%random_str的最大连续字符数必须大于0。")
        if not source.text:
            raise SchemaValidationError(f"%random_str的字符串来源不能为空：{source_key}。")
        length = random_source.randint(1, min(maximum, len(source.text)))
        start = random_source.randrange(len(source.text) - length + 1)
        return source.text[start : start + length]

    rendered = _RANDOM_STRING.sub(replace, value)
    if "%random_str" in rendered:
        raise SchemaValidationError(
            "%random_str必须写为%random_str($字符串来源, 最大连续字符数)。"
        )
    return rendered


def _render_template_value(
    value: str, values: dict[str, RenderedValue], random_source: random.Random
) -> str:
    """按固定顺序展开字符串和数字随机模板函数。"""
    return _render_random_integers(_render_random_strings(value, values, random_source), random_source)


def _render_name(
    template: str,
    values: dict[str, RenderedValue],
    optional_enabled: dict[str, bool],
) -> tuple[str, list[dict[str, object]]]:
    """按模板替换占位符并生成全局原文偏移。"""
    pieces: list[str] = []
    spans: list[dict[str, object]] = []
    cursor = 0
    offset = 0
    for match in _PLACEHOLDER.finditer(template):
        literal = template[cursor : match.start()]
        pieces.append(literal)
        offset += len(literal)
        field = match.group("field")
        index = match.group("index")
        value_key = f"{field}[{index}]" if index is not None else field
        if value_key not in values:
            raise SchemaValidationError(f"模板使用了未定义字段：{field}。")
        if match.group("marker") == "&" and not optional_enabled[field]:
            cursor = match.end()
            continue
        rendered = values[value_key]
        pieces.append(rendered.text)
        for start, end, label in rendered.spans:
            spans.append(
                {
                    "start": offset + start,
                    "end": offset + end,
                    "text": rendered.text[start:end],
                    "label": label,
                    "source": "synthetic_template",
                }
            )
        offset += len(rendered.text)
        cursor = match.end()
    suffix = template[cursor:]
    pieces.append(suffix)
    text = "".join(pieces)
    return text, spans


def _bio_labels(text: str, spans: list[dict[str, object]]) -> list[str]:
    """把不重叠片段转换为字符级 BIO 标签。"""
    labels = ["O"] * len(text)
    previous_end = 0
    for span in sorted(spans, key=lambda item: (int(item["start"]), int(item["end"]))):
        start, end, label = int(span["start"]), int(span["end"]), str(span["label"])
        if not 0 <= start < end <= len(text) or text[start:end] != span["text"]:
            raise SchemaValidationError("合成片段的位置或文本无效。")
        if start < previous_end:
            raise SchemaValidationError("合成模板产生了重叠片段。")
        labels[start] = f"B-{label}"
        labels[start + 1 : end] = [f"I-{label}"] * (end - start - 1)
        previous_end = end
    if any(label not in BIO_LABELS for label in labels):
        raise SchemaValidationError("合成模板产生了未知BIO标签。")
    return labels


def iter_synthetic_records(
    payload: dict[str, Any],
    *,
    count: int,
    seed: int = 20260918,
    optional_rate: float = 1.0,
    season_rate: float = 0.7,
) -> Iterable[dict[str, object]]:
    """从词表和名称模板确定性抽样，生成仅供训练集使用的字符级 BIO 记录。"""
    if count <= 0:
        raise InputValidationError("生成数量必须大于0。")
    if not 0.0 <= optional_rate <= 1.0:
        raise InputValidationError("可选字段填充比例必须在0到1之间。")
    if not 0.0 <= season_rate <= 1.0:
        raise InputValidationError("含季数标题的抽样比例必须在0到1之间。")
    templates = payload["name_template"]
    titles = payload["title"]
    if not templates or not titles:
        raise SchemaValidationError("模板和标题列表不能为空。")
    title_items = [item for item in titles if isinstance(item, dict)]
    if len(title_items) != len(titles):
        raise SchemaValidationError("title列表中的每项必须是对象。")
    _validate_title_boundaries(templates, title_items)
    random_source = random.Random(seed)
    eligibility_cache: dict[tuple[int, int], tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for index in range(count):
        template_group_index = random_source.randrange(len(templates))
        template_group = templates[template_group_index]
        if not isinstance(template_group, dict):
            raise SchemaValidationError("name_template中的条目必须是对象。")
        names = template_group.get("name")
        if not isinstance(names, list) or not names:
            raise SchemaValidationError("name_template中的name必须是非空列表。")
        template_index = random_source.randrange(len(names))
        template = str(names[template_index])
        placeholders = list(_PLACEHOLDER.finditer(template))
        template_fields = {match.group("field") for match in placeholders}
        alias_indexes = [
            int(match.group("index"))
            for match in placeholders
            if match.group("field") == "title_alias" and match.group("index") is not None
        ]
        if any(item <= 0 for item in alias_indexes):
            raise SchemaValidationError("title_alias索引从1开始。")
        required_alias_count = max(alias_indexes, default=0)
        exclude_title_season = template_group.get("exclude_title_season_expr", False)
        if not isinstance(exclude_title_season, bool):
            raise SchemaValidationError("exclude_title_season_expr必须为布尔值。")
        cache_key = (template_group_index, template_index)
        if cache_key not in eligibility_cache:
            eligible_titles = [
                item
                for item in title_items
                if len(_aliases(item)) >= required_alias_count and not _template_is_disabled(template_group, item)
            ]
            for field in template_fields & _TITLE_FIELD_LABELS.keys():
                eligible_titles = [item for item in eligible_titles if _template_title_value(item, field, _aliases(item)).strip()]
            if exclude_title_season:
                eligible_titles = [item for item in eligible_titles if not any(str(value) for value in item.get("season_expr", []))]
            if not eligible_titles:
                raise SchemaValidationError("当前模板条件下没有可用标题条目。")
            season_titles = [item for item in eligible_titles if any(str(value) for value in item.get("season_expr", []))]
            ordinary_titles = [item for item in eligible_titles if item not in season_titles]
            eligibility_cache[cache_key] = (season_titles, ordinary_titles)
        season_titles, ordinary_titles = eligibility_cache[cache_key]
        if season_titles and (not ordinary_titles or random_source.random() < season_rate):
            title_item = random_source.choice(season_titles)
        else:
            title_item = random_source.choice(ordinary_titles or season_titles)
        title_seasons = [str(item) for item in title_item.get("season_expr", []) if str(item)]
        template_seasons = [str(item) for item in template_group.get("season_expr", []) if str(item)]
        if exclude_title_season:
            if "season_expr" in {match.group("field") for match in placeholders} and not template_seasons:
                raise SchemaValidationError("排除标题季数的模板必须提供非空season_expr列表。")
            seasons = [
                _render_random_integers(random_source.choice(template_seasons), random_source)
            ] if template_seasons else []
        else:
            seasons = title_seasons
        aliases = _aliases(title_item)
        alias_open = str(template_group.get("title_alias_open", ""))
        alias_begin = str(template_group.get("title_alias_begin", ""))
        title_en = _english_title(title_item, aliases)
        title_cn = _chinese_title(title_item)
        legacy_title = _title_text(title_item, "title")
        values: dict[str, RenderedValue] = {
            "title": _title_value(legacy_title, "TITLE", title_seasons),
            "title_en": _title_value(title_en, "TITLE", title_seasons),
            "title_cn": _title_value(title_cn, "TITLE_ALIAS", title_seasons),
            "title_as_title_alias_1": _title_value(
                legacy_title, "TITLE_ALIAS", title_seasons
            ),
            "title_alias": _title_alias_value(
                aliases,
                open_text=alias_open,
                begin_text=alias_begin,
                close_text=str(template_group.get("title_alias_close", "")),
                separator=str(template_group.get("title_alias_separator", "")),
                season_values=title_seasons,
            ),
            "season_expr": _ordinary_value(random_source.choice(seasons), "SEASON_EXPR")
            if seasons
            else RenderedValue("", ()),
        }
        for alias_index in range(1, required_alias_count + 1):
            values[f"title_alias[{alias_index}]"] = _title_alias_value(
                [aliases[alias_index - 1]],
                open_text=alias_open,
                begin_text=alias_begin,
                close_text=str(template_group.get("title_alias_close", "")),
                separator=str(template_group.get("title_alias_separator", "")),
                season_values=title_seasons,
            )
        if "random_month" in {match.group("field") for match in placeholders}:
            months = [str(item) for item in template_group.get("random_month", []) if str(item)]
            if not months:
                raise SchemaValidationError("使用random_month的模板必须提供非空random_month列表。")
            values["random_month"] = RenderedValue(
                _render_template_value(random_source.choice(months), values, random_source), ()
            )
        for field, label in _TEMPLATE_LOCAL_FIELD_LABELS.items():
            if field not in template_fields:
                continue
            field_values = [
                str(item) for item in template_group.get(field, []) if str(item)
            ]
            if not field_values:
                raise SchemaValidationError(
                    f"使用{field}的模板必须在同一模板组提供非空{field}列表。"
                )
            value = _render_template_value(random_source.choice(field_values), values, random_source)
            values[field] = _ordinary_value(value, label)
        for field, label in _FIELD_LABELS.items():
            if field in values or field == "season_expr":
                continue
            value = _render_template_value(
                str(random_source.choice(payload[field])), values, random_source
            )
            values[field] = _ordinary_value(value, label)
        optional_enabled = {field: random_source.random() < optional_rate for field in values}
        text, spans = _render_name(template, values, optional_enabled)
        labels = _bio_labels(text, spans)
        digest = hashlib.sha256(f"{seed}:{index}:{template}:{text}".encode("utf-8")).hexdigest()
        yield {
                "sample_id": f"synthetic-{digest[:24]}",
                "split": "train",
                "text": text,
                "characters": list(text),
                "labels": labels,
                "spans": spans,
                "schema_version": "1.0",
                "source": "synthetic_template",
                "template": template,
                "template_group_id": f"synthetic-template-group-{template_group_index + 1:02d}",
                "template_entry_id": (
                    f"synthetic-template-group-{template_group_index + 1:02d}"
                    f"-entry-{template_index + 1:02d}"
                ),
                "work_key": _work_key(title_item),
            }


def generate_synthetic_records(
    payload: dict[str, Any],
    *,
    count: int,
    seed: int = 20260918,
    optional_rate: float = 1.0,
    season_rate: float = 0.7,
) -> list[dict[str, object]]:
    """保留小数据调用的列表接口，大数据使用迭代接口。"""
    return list(iter_synthetic_records(payload, count=count, seed=seed, optional_rate=optional_rate, season_rate=season_rate))


def write_synthetic_records(records: Iterable[dict[str, object]], output: str | Path) -> dict[str, object]:
    """写入 JSONL 及旁路清单，防止合成数据被误作人工金标。"""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for item in records:
            line = json.dumps(item, ensure_ascii=False, sort_keys=True)
            stream.write(line + "\n")
            digest.update((line + "\n").encode("utf-8"))
            count += 1
    manifest = {
        "版本": 1,
        "格式": "合成字符级BIO JSONL",
        "训练用途": "仅限训练分区，不得作为验证、测试或人工金标",
        "样本数": count,
        "SHA256": digest.hexdigest(),
        "文件": str(target),
    }
    target.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def synthetic_work_key(record: dict[str, object]) -> str:
    """从主标题片段生成稳定作品键，确保同一作品不跨训练分区。"""
    explicit_key = str(record.get("work_key", "")).strip().casefold()
    if explicit_key:
        return explicit_key
    title_parts = [
        str(span["text"])
        for span in record.get("spans", [])
        if isinstance(span, dict) and span.get("label") == "TITLE"
    ]
    key = "".join(title_parts).strip().casefold()
    if not key:
        raise SchemaValidationError("合成记录缺少可用于划分的作品键或主标题片段。")
    return key


def synthetic_template_group_key(record: dict[str, object]) -> str:
    """取得稳定的模板组键；旧产物缺少组ID时退化为单模板键。"""
    template_group_id = str(record.get("template_group_id", "")).strip()
    if template_group_id:
        return template_group_id
    template = str(record.get("template", "")).strip()
    if not template:
        raise SchemaValidationError("合成记录缺少可用于划分的模板组或模板文本。")
    return "legacy-template-" + hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


def synthetic_template_entry_key(record: dict[str, object]) -> str:
    """取得模板条目的稳定键，区分文本相同但来源条目不同的模板。"""
    entry_id = str(record.get("template_entry_id", "")).strip()
    if entry_id:
        return entry_id
    return synthetic_template_group_key(record) + ":" + hashlib.sha256(
        str(record.get("template", "")).encode("utf-8")
    ).hexdigest()[:16]


def _template_split_assignments(template_keys: set[str], *, seed: int) -> dict[str, str]:
    """将模板组整体分配到三个分区，保证各分区至少保留一个模板组。"""
    if len(template_keys) < 3:
        raise SchemaValidationError("模板双隔离至少需要三个不同的模板组。")
    ordered = sorted(
        template_keys,
        key=lambda key: hashlib.sha256(f"{seed}:template:{key}".encode("utf-8")).hexdigest(),
    )
    validation_count = max(1, len(ordered) // 10)
    test_count = max(1, len(ordered) // 10)
    train_count = len(ordered) - validation_count - test_count
    if train_count < 1:
        raise SchemaValidationError("模板组数量不足，无法同时生成训练、验证和测试分区。")
    assignments: dict[str, str] = {}
    for index, key in enumerate(ordered):
        assignments[key] = (
            "train"
            if index < train_count
            else "validation"
            if index < train_count + validation_count
            else "test"
        )
    return assignments


def _work_split_assignments(work_keys: set[str], *, seed: int) -> dict[str, str]:
    """按稳定哈希排序后均衡分配作品，避免小分区因随机阈值异常缩水。"""
    if len(work_keys) < 3:
        raise SchemaValidationError("作品双隔离至少需要三个不同的主标题条目。")
    ordered = sorted(
        work_keys,
        key=lambda key: hashlib.sha256(f"{seed}:work:{key}".encode("utf-8")).hexdigest(),
    )
    validation_count = max(1, round(len(ordered) * 0.1))
    test_count = max(1, round(len(ordered) * 0.1))
    train_count = len(ordered) - validation_count - test_count
    if train_count < 1:
        raise SchemaValidationError("作品条目数不足，无法同时生成训练、验证和测试分区。")
    assignments: dict[str, str] = {}
    for index, key in enumerate(ordered):
        assignments[key] = (
            "train"
            if index < train_count
            else "validation"
            if index < train_count + validation_count
            else "test"
        )
    return assignments


def split_synthetic_records(
    records: Iterable[dict[str, object]],
    *,
    seed: int = 20260918,
    template_coverage: bool = False,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    """按作品确定性划分合成记录，可要求每个分区覆盖全部模板条目。"""
    source_records = list(records)
    splits: dict[str, list[dict[str, object]]] = {"train": [], "validation": [], "test": []}
    work_keys = {synthetic_work_key(record) for record in source_records}
    work_splits = _work_split_assignments(work_keys, seed=seed)
    template_splits = (
        None
        if template_coverage
        else _template_split_assignments(
            {synthetic_template_group_key(record) for record in source_records}, seed=seed
        )
    )
    seen_ids: set[str] = set()
    excluded_count = 0
    for record in source_records:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id in seen_ids:
            raise SchemaValidationError("合成记录样本ID为空或重复。")
        seen_ids.add(sample_id)
        key = synthetic_work_key(record)
        split = work_splits[key]
        template_key = synthetic_template_group_key(record)
        if template_splits is not None and split != template_splits[template_key]:
            excluded_count += 1
            continue
        item = dict(record)
        item["split"] = split
        item["work_group_id"] = "synthetic-work-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        item["template_group_id"] = template_key
        splits[split].append(item)
    template_entry_keys = {synthetic_template_entry_key(record) for record in source_records}
    split_template_entries = {
        name: {synthetic_template_entry_key(record) for record in items}
        for name, items in splits.items()
    }
    if template_coverage:
        missing = {
            name: sorted(template_entry_keys - entries)
            for name, entries in split_template_entries.items()
            if template_entry_keys - entries
        }
        if missing:
            details = "；".join(f"{name}缺少{','.join(values)}" for name, values in missing.items())
            raise SchemaValidationError(f"作品隔离后的分区未覆盖全部模板条目：{details}。")
    report = {
        "版本": 1,
        "随机种子": seed,
        "样本数": len(seen_ids),
        "作品条目数": len(work_splits),
        "分区样本数": {name: len(items) for name, items in splits.items()},
        "分区作品数": {
            name: sum(1 for assigned in work_splits.values() if assigned == name)
            for name in splits
        },
        "模板组数": len({synthetic_template_group_key(record) for record in source_records}),
        "分区模板组数": {
            name: len({synthetic_template_group_key(record) for record in items})
            for name, items in splits.items()
        },
        "模板条目数": len(template_entry_keys),
        "分区模板条目数": {name: len(entries) for name, entries in split_template_entries.items()},
        "模板组分配": dict(sorted(template_splits.items())) if template_splits is not None else {},
        "双隔离排除样本数": excluded_count,
        "划分策略": "作品隔离且每个分区覆盖全部模板条目" if template_coverage else "作品和模板组双隔离",
        "隔离依据": (
            "主标题片段按稳定分配隔离；每个分区保留全部模板条目，模板结构会在分区间重复。"
            if template_coverage
            else "主标题片段与模板组均经过稳定分配；只有二者归属同一分区的样本才会导出。"
        ),
    }
    return splits, report


def write_synthetic_splits(
    splits: dict[str, list[dict[str, object]]],
    report: dict[str, object],
    output_dir: str | Path,
) -> dict[str, object]:
    """写入模板训练、验证、测试分区和可复现清单。"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    for split in ("train", "validation", "test"):
        target = directory / f"{split}.jsonl"
        digest = hashlib.sha256()
        with target.open("w", encoding="utf-8", newline="\n") as stream:
            for item in splits[split]:
                line = json.dumps(item, ensure_ascii=False, sort_keys=True)
                stream.write(line + "\n")
                digest.update((line + "\n").encode("utf-8"))
        files[split] = {"文件": str(target), "样本数": len(splits[split]), "SHA256": digest.hexdigest()}
    manifest = {
        **report,
        "格式": "模板合成字符级BIO JSONL",
        "标签": list(BIO_LABELS),
        "分区": files,
    }
    (directory / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def audit_synthetic_records(records: Iterable[dict[str, object]]) -> dict[str, object]:
    """统计模板、字段和跨批重复情况，不改写任何合成样本。"""
    items = list(records)
    ids = [str(item.get("sample_id", "")) for item in items]
    texts = [str(item.get("text", "")) for item in items]
    template_counts = Counter(str(item.get("template", "")) for item in items)
    group_counts = Counter(str(item.get("template_group_id", "")) for item in items)
    entry_counts = Counter(synthetic_template_entry_key(item) for item in items)
    field_counts: Counter[str] = Counter()
    invalid_count = 0
    for item in items:
        try:
            if list(item.get("characters", [])) != list(str(item.get("text", ""))) or list(item.get("labels", [])) != _bio_labels(str(item.get("text", "")), list(item.get("spans", []))):
                invalid_count += 1
        except (KeyError, TypeError, ValueError, SchemaValidationError):
            invalid_count += 1
        field_counts.update(str(span.get("label", "")) for span in item.get("spans", []) if isinstance(span, dict))
    return {
        "样本数": len(items),
        "唯一样本ID数": len(set(ids)),
        "重复样本ID数": len(ids) - len(set(ids)),
        "唯一文本数": len(set(texts)),
        "重复文本数": len(texts) - len(set(texts)),
        "BIO不一致样本数": invalid_count,
        "模板数": len(template_counts),
        "模板组数": len(group_counts),
        "模板条目数": len(entry_counts),
        "模板覆盖": dict(sorted(template_counts.items())),
        "模板组覆盖": dict(sorted(group_counts.items())),
        "模板条目覆盖": dict(sorted(entry_counts.items())),
        "字段覆盖": dict(sorted(field_counts.items())),
    }


def deduplicate_synthetic_records(records: Iterable[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, int]]:
    """按原文和完整字符BIO标签去重，拒绝同文不同标签的冲突样本。"""
    kept: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_texts: dict[str, tuple[str, ...]] = {}
    duplicate_ids = duplicate_texts = 0
    for item in records:
        sample_id = str(item.get("sample_id", ""))
        text = str(item.get("text", ""))
        labels = tuple(map(str, item.get("labels", [])))
        if not sample_id or not text or not labels:
            raise SchemaValidationError("去重输入包含缺少样本ID、文本或BIO标签的记录。")
        if sample_id in seen_ids:
            duplicate_ids += 1
            continue
        previous_labels = seen_texts.get(text)
        if previous_labels is not None:
            if previous_labels != labels:
                raise SchemaValidationError("同一合成文本出现了不同的BIO标签，无法自动去重。")
            duplicate_texts += 1
            continue
        seen_ids.add(sample_id)
        seen_texts[text] = labels
        kept.append(item)
    return kept, {"重复样本ID数": duplicate_ids, "重复文本数": duplicate_texts, "保留样本数": len(kept)}
