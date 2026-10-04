"""按配置顺序运行名称预处理规则。"""

from __future__ import annotations

from pathlib import Path

from anitopy_ml.errors import PreprocessingError
from anitopy_ml.processing.config import resolve_processing_config
from anitopy_ml.processing.mapping import apply_edits
from anitopy_ml.processing.rules import edits_for_rule
from anitopy_ml.processing.types import (
    PreprocessingResult,
    ProcessingConfig,
    ProcessingStep,
    RuleMatchSummary,
    TextState,
)


class PreprocessingPipeline:
    """持有不可变配置并为每个名称创建独立处理状态。"""

    def __init__(self, config: ProcessingConfig | str | Path | None = None) -> None:
        self.config = resolve_processing_config(config)

    def apply(
        self,
        title: str,
        *,
        trace: bool = False,
        max_output_length: int | None = None,
    ) -> PreprocessingResult:
        """返回处理后文本、来源映射及每条规则的命中数。"""
        if not isinstance(title, str) or not title.strip():
            raise PreprocessingError("标题不能为空。", code="PREPROCESSING_EMPTY")
        limit = (
            self.config.max_output_length
            if max_output_length is None
            else min(self.config.max_output_length, max_output_length)
        )
        if not self.config.enabled:
            if len(title) > limit:
                raise PreprocessingError(
                    f"名称长度超过调用方上限{limit}个字符。",
                    code="PREPROCESSING_LIMIT",
                )
            state = TextState.from_raw(title)
            return PreprocessingResult(
                raw_text=title,
                processed_text=title,
                mappings=state.mappings,
                config_version=self.config.config_version,
                config_fingerprint=self.config.fingerprint,
                matched_rules=(),
                changed=False,
            )

        if len(title) > limit:
            raise PreprocessingError(
                f"名称长度超过预处理上限{limit}个字符。",
                code="PREPROCESSING_LIMIT",
            )
        state = TextState.from_raw(title)
        matched_rules: list[RuleMatchSummary] = []
        steps: list[ProcessingStep] = []
        for rule in self.config.rules:
            if not rule.enabled:
                continue
            before = state.text
            edits = edits_for_rule(rule, before)
            resulting_length = len(before) + sum(
                len(edit.replacement) - (edit.end - edit.start)
                for edit in edits
            )
            if resulting_length > limit:
                raise PreprocessingError(
                    f"规则{rule.id}的处理结果超过预处理上限{limit}个字符。",
                    code="PREPROCESSING_LIMIT",
                )
            state = apply_edits(state, edits)
            matched_rules.append(RuleMatchSummary(rule.id, len(edits)))
            if trace:
                steps.append(ProcessingStep(rule.id, before, state.text, len(edits)))

        if not state.text.strip():
            raise PreprocessingError(
                "预处理后名称为空，无法继续解析。",
                code="PREPROCESSING_EMPTY",
            )
        return PreprocessingResult(
            raw_text=title,
            processed_text=state.text,
            mappings=state.mappings,
            config_version=self.config.config_version,
            config_fingerprint=self.config.fingerprint,
            matched_rules=tuple(matched_rules),
            changed=state.text != title,
            steps=tuple(steps),
        )
