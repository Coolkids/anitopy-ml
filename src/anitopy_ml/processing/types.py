"""名称处理流水线使用的不可变配置与结果类型。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from re import Pattern
from typing import Literal

from anitopy_ml.normalizer import SourceRange


RuleType = Literal["string_replace", "regex_replace", "string_mask", "regex_mask"]


@dataclass(frozen=True, slots=True)
class ReplacementToken:
    """正则替换模板中的字面量或捕获组引用。"""

    literal: str | None = None
    group: str | int | None = None


@dataclass(frozen=True, slots=True)
class ProcessingRule:
    """已经校验并编译的单条有序名称处理规则。"""

    id: str
    enabled: bool
    type: RuleType
    match: str | None = None
    pattern_text: str | None = None
    pattern: Pattern[str] | None = None
    replacement: str | None = None
    replacement_tokens: tuple[ReplacementToken, ...] = ()
    count: int = 0


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    """不可变预处理设置及稳定配置指纹。"""

    config_version: int
    enabled: bool
    max_output_length: int
    rules: tuple[ProcessingRule, ...]
    fingerprint: str
    source_path: str | None = None

    @classmethod
    def build(
        cls,
        *,
        config_version: int,
        enabled: bool,
        max_output_length: int,
        rules: tuple[ProcessingRule, ...],
        source_path: str | None = None,
    ) -> "ProcessingConfig":
        """按规范化后的有效配置生成稳定指纹。"""
        serializable = {
            "config_version": config_version,
            "preprocessing": {
                "enabled": enabled,
                "max_output_length": max_output_length,
                "rules": [
                    {
                        "id": rule.id,
                        "enabled": rule.enabled,
                        "type": rule.type,
                        "match": rule.match,
                        "pattern": rule.pattern_text,
                        "replacement": rule.replacement,
                        "flags": sorted(
                            flag
                            for flag, value in (
                                ("IGNORECASE", 2),
                                ("MULTILINE", 8),
                                ("DOTALL", 16),
                                ("ASCII", 256),
                            )
                            if rule.pattern is not None and rule.pattern.flags & value
                        ),
                        "count": rule.count,
                    }
                    for rule in rules
                ],
            },
            "result_processing": {"enabled": False, "rules": []},
        }
        encoded = json.dumps(serializable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return cls(
            config_version=config_version,
            enabled=enabled,
            max_output_length=max_output_length,
            rules=rules,
            fingerprint=fingerprint,
            source_path=source_path,
        )


@dataclass(frozen=True, slots=True)
class TextEdit:
    """相对于某一步输入文本的替换区间。"""

    start: int
    end: int
    replacement: str


@dataclass(frozen=True, slots=True)
class TextState:
    """当前名称及每个字符所能追溯到的原文区间。"""

    raw_text: str
    text: str
    mappings: tuple[tuple[SourceRange, ...], ...]

    @classmethod
    def from_raw(cls, text: str) -> "TextState":
        """创建一对一初始来源映射。"""
        return cls(
            raw_text=text,
            text=text,
            mappings=tuple((SourceRange(index, index + 1),) for index in range(len(text))),
        )


@dataclass(frozen=True, slots=True)
class RuleMatchSummary:
    """一条已启用规则的命中数量。"""

    id: str
    count: int


@dataclass(frozen=True, slots=True)
class ProcessingStep:
    """调试预览中的单步输入、输出与命中数量。"""

    id: str
    before: str
    after: str
    matched: int


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    """处理后文本、原文映射和可公开的处理摘要。"""

    raw_text: str
    processed_text: str
    mappings: tuple[tuple[SourceRange, ...], ...]
    config_version: int
    config_fingerprint: str
    matched_rules: tuple[RuleMatchSummary, ...]
    changed: bool
    steps: tuple[ProcessingStep, ...] = ()


@dataclass(frozen=True, slots=True)
class ProcessingMetadata:
    """可选解析结果摘要，不包含内部字符映射。"""

    config_version: int
    config_fingerprint: str
    processed_text: str
    changed: bool
    matched_rules: tuple[RuleMatchSummary, ...]

    def model_dump(self) -> dict[str, object]:
        """生成只含 JSON 兼容值的公开摘要。"""
        return {
            "config_version": self.config_version,
            "config_fingerprint": self.config_fingerprint,
            "processed_text": self.processed_text,
            "changed": self.changed,
            "matched_rules": [
                {"id": item.id, "count": item.count} for item in self.matched_rules
            ],
        }
