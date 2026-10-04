"""读取、严格校验并编译媒体名称处理 TOML 配置。"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anitopy_ml.errors import ConfigurationError
from anitopy_ml.processing.types import (
    ProcessingConfig,
    ProcessingRule,
    ReplacementToken,
)

_RULE_TYPES = {"string_replace", "regex_replace", "string_mask", "regex_mask"}
_REGEX_FLAGS = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
    "ASCII": re.ASCII,
}
_ROOT_KEYS = {"config_version", "preprocessing", "result_processing"}


@dataclass(frozen=True, slots=True)
class ConfigurationReport:
    """无需加载模型即可显示的配置检查结果。"""

    config: ProcessingConfig
    warnings: tuple[str, ...] = ()


def _fail(location: str, message: str) -> ConfigurationError:
    """统一加入 TOML 配置字段路径。"""
    return ConfigurationError(f"配置字段{location}：{message}")


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise _fail(location, "必须是表。")
    return value


def _keys(table: Mapping[str, Any], allowed: set[str], required: set[str], location: str) -> None:
    unknown = set(table) - allowed
    missing = required - set(table)
    if unknown:
        raise _fail(location, f"包含未知字段：{', '.join(sorted(unknown))}。")
    if missing:
        raise _fail(location, f"缺少必填字段：{', '.join(sorted(missing))}。")


def _boolean(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        raise _fail(location, "必须是布尔值。")
    return value


def _integer(value: object, location: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise _fail(location, f"必须是{minimum}到{maximum}之间的整数。")
    return value


def _bounded_text(value: object, location: str, maximum: int, *, nonempty: bool) -> str:
    if not isinstance(value, str) or len(value) > maximum or (nonempty and not value):
        rule = "非空字符串" if nonempty else "字符串"
        raise _fail(location, f"必须是长度不超过{maximum}的{rule}。")
    return value


def _parse_replacement(template: str, pattern: re.Pattern[str], location: str) -> tuple[ReplacementToken, ...]:
    r"""解析受限的字面量与 \g<组名> 替换模板。"""
    tokens: list[ReplacementToken] = []
    literal: list[str] = []

    def flush() -> None:
        if literal:
            tokens.append(ReplacementToken(literal="".join(literal)))
            literal.clear()

    index = 0
    while index < len(template):
        character = template[index]
        if character != "\\":
            literal.append(character)
            index += 1
            continue
        if index + 1 < len(template) and template[index + 1] == "\\":
            literal.append("\\")
            index += 2
            continue
        if not template.startswith("\\g<", index):
            raise _fail(location, "正则替换只支持\\g<组名>、\\g<组号>和\\\\转义。")
        end = template.find(">", index + 3)
        if end < 0:
            raise _fail(location, "捕获组引用缺少右尖括号。")
        name = template[index + 3 : end]
        if not name:
            raise _fail(location, "捕获组引用不能为空。")
        if name.isdecimal():
            group: str | int = int(name)
        else:
            group = name
        try:
            pattern.groupindex[group] if isinstance(group, str) else None
            if isinstance(group, int) and not 0 <= group <= pattern.groups:
                raise IndexError(group)
        except (KeyError, IndexError) as error:
            raise _fail(location, f"替换模板引用了不存在的捕获组：{name}。") from error
        flush()
        tokens.append(ReplacementToken(group=group))
        index = end + 1
    flush()
    return tuple(tokens)


def _compile_rule(value: object, index: int, identifiers: set[str]) -> ProcessingRule:
    location = f"preprocessing.rules[{index}]"
    table = _mapping(value, location)
    rule_id = _bounded_text(table.get("id"), f"{location}.id", 128, nonempty=True)
    if not rule_id.strip():
        raise _fail(f"{location}.id", "不能只包含空白字符。")
    if rule_id in identifiers:
        raise _fail(f"{location}.id", "规则标识重复。")
    identifiers.add(rule_id)

    kind = table.get("type")
    if not isinstance(kind, str) or kind not in _RULE_TYPES:
        raise _fail(f"{location}.type", f"必须是以下类型之一：{', '.join(sorted(_RULE_TYPES))}。")
    is_regex = kind.startswith("regex_")
    is_replace = kind.endswith("replace")
    allowed = {"id", "enabled", "type", "count"}
    allowed.add("pattern" if is_regex else "match")
    if is_replace:
        allowed.add("replacement")
    if is_regex:
        allowed.add("flags")
    required = {"id", "type", "pattern" if is_regex else "match"}
    if is_replace:
        required.add("replacement")
    _keys(table, allowed, required, location)

    enabled = _boolean(table.get("enabled", True), f"{location}.enabled")
    count = _integer(table.get("count", 0), f"{location}.count", 0, 1_000_000)
    match = None
    pattern_text = None
    compiled = None
    tokens: tuple[ReplacementToken, ...] = ()
    replacement = None

    if is_regex:
        pattern_text = _bounded_text(table.get("pattern"), f"{location}.pattern", 1024, nonempty=True)
        raw_flags = table.get("flags", [])
        if not isinstance(raw_flags, list) or any(not isinstance(flag, str) for flag in raw_flags):
            raise _fail(f"{location}.flags", "必须是正则选项名称数组。")
        if len(set(raw_flags)) != len(raw_flags):
            raise _fail(f"{location}.flags", "不能重复指定正则选项。")
        unknown_flags = set(raw_flags) - set(_REGEX_FLAGS)
        if unknown_flags:
            raise _fail(f"{location}.flags", f"未知选项：{', '.join(sorted(unknown_flags))}。")
        combined_flags = re.NOFLAG
        for flag in raw_flags:
            combined_flags |= _REGEX_FLAGS[flag]
        try:
            compiled = re.compile(pattern_text, combined_flags)
        except re.error as error:
            raise _fail(f"{location}.pattern", f"正则表达式无效：{error}。") from error
        empty = compiled.match("")
        if empty is not None and empty.start() == empty.end():
            raise _fail(f"{location}.pattern", "不能匹配空字符串。")
    else:
        match = _bounded_text(table.get("match"), f"{location}.match", 1024, nonempty=True)

    if is_replace:
        replacement = _bounded_text(table.get("replacement"), f"{location}.replacement", 4096, nonempty=False)
        if is_regex:
            assert compiled is not None
            tokens = _parse_replacement(replacement, compiled, f"{location}.replacement")

    return ProcessingRule(
        id=rule_id,
        enabled=enabled,
        type=kind,
        match=match,
        pattern_text=pattern_text,
        pattern=compiled,
        replacement=replacement,
        replacement_tokens=tokens,
        count=count,
    )


def parse_processing_config(payload: object, *, source_path: str | None = None) -> ProcessingConfig:
    """严格解析内存中的 TOML 解码对象。"""
    root = _mapping(payload, "顶层")
    _keys(root, _ROOT_KEYS, {"config_version"}, "顶层")
    version = _integer(root["config_version"], "config_version", 1, 1)

    preprocessing = _mapping(root.get("preprocessing", {}), "preprocessing")
    _keys(
        preprocessing,
        {"enabled", "max_output_length", "rules"},
        set(),
        "preprocessing",
    )
    enabled = _boolean(preprocessing.get("enabled", False), "preprocessing.enabled")
    max_length = _integer(
        preprocessing.get("max_output_length", 16384),
        "preprocessing.max_output_length",
        1,
        16384,
    )
    raw_rules = preprocessing.get("rules", [])
    if not isinstance(raw_rules, list):
        raise _fail("preprocessing.rules", "必须是规则数组。")
    if len(raw_rules) > 100:
        raise _fail("preprocessing.rules", "规则数量不能超过100条。")
    identifiers: set[str] = set()
    rules = tuple(_compile_rule(rule, index, identifiers) for index, rule in enumerate(raw_rules))

    result_processing = _mapping(root.get("result_processing", {}), "result_processing")
    _keys(result_processing, {"enabled", "rules"}, set(), "result_processing")
    result_enabled = _boolean(result_processing.get("enabled", False), "result_processing.enabled")
    result_rules = result_processing.get("rules", [])
    if not isinstance(result_rules, list):
        raise _fail("result_processing.rules", "必须是数组。")
    if result_enabled or result_rules:
        raise _fail("result_processing", "本版本只预留透传入口，必须保持关闭且规则为空。")

    return ProcessingConfig.build(
        config_version=version,
        enabled=enabled,
        max_output_length=max_length,
        rules=rules,
        source_path=source_path,
    )


def load_processing_config(path: str | Path) -> ProcessingConfig:
    """读取 UTF-8 TOML 文件并验证配置；显式路径错误不会回退。"""
    resolved = Path(path).expanduser().resolve()
    try:
        with resolved.open("rb") as stream:
            payload = tomllib.load(stream)
    except FileNotFoundError as error:
        raise ConfigurationError(f"预处理配置文件不存在：{resolved}。") from error
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(f"无法读取预处理配置文件{resolved}：{error}。") from error
    try:
        return parse_processing_config(payload, source_path=str(resolved))
    except ConfigurationError as error:
        raise ConfigurationError(f"配置文件{resolved}：{error}") from error


def disabled_processing_config() -> ProcessingConfig:
    """创建不执行名称规则的内置默认配置。"""
    return ProcessingConfig.build(
        config_version=1,
        enabled=False,
        max_output_length=16384,
        rules=(),
    )


def resolve_processing_config(
    value: ProcessingConfig | str | Path | None,
) -> ProcessingConfig:
    """解析库调用显式传入的配置；None 表示关闭且不读取环境变量。"""
    if value is None:
        return disabled_processing_config()
    if isinstance(value, ProcessingConfig):
        return value
    if isinstance(value, (str, Path)):
        return load_processing_config(value)
    raise ConfigurationError("processing_config必须是路径、ProcessingConfig或None。")


def inspect_processing_config(path: str | Path) -> ConfigurationReport:
    """加载并返回可供命令行展示的配置检查报告。"""
    return ConfigurationReport(config=load_processing_config(path))
