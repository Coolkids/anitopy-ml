"""将受约束 BIO 预测还原为带原文证据的解析结果。"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Sequence

from anitopy_ml.constraints import extract_constraints
from anitopy_ml.errors import SchemaValidationError
from anitopy_ml.schemas import BIO_LABELS, Evidence, ParseResult, Span


def spans_from_bio(
    raw_text: str,
    labels: Sequence[str],
    confidences: Sequence[float] | None = None,
) -> tuple[Span, ...]:
    """把逐字符 BIO 标签转换为原文片段，拒绝越界和非法转移。"""
    if len(raw_text) != len(labels):
        raise SchemaValidationError("BIO标签数量必须与原始标题字符数一致。")
    if confidences is not None and len(confidences) != len(labels):
        raise SchemaValidationError("BIO置信度数量必须与标签数量一致。")

    spans: list[Span] = []
    start: int | None = None
    entity_type: str | None = None

    def close(end: int) -> None:
        nonlocal start, entity_type
        if start is not None and entity_type is not None:
            spans.append(
                Span(
                    start=start,
                    end=end,
                    text=raw_text[start:end],
                    label=entity_type,  # 标签已由固定 BIO 标签表校验。
                    source="model",
                )
            )
        start, entity_type = None, None

    for index, label in enumerate(labels):
        if label not in BIO_LABELS:
            raise SchemaValidationError("BIO预测包含未知标签。")
        if label == "O":
            close(index)
            continue
        prefix, current_type = label.split("-", maxsplit=1)
        if prefix == "B":
            close(index)
            start, entity_type = index, current_type
            continue
        if prefix != "I" or start is None or entity_type != current_type:
            raise SchemaValidationError("BIO预测存在未被约束解码修复的非法内部标签。")
    close(len(raw_text))
    return tuple(spans)


def _mean_confidence(
    span: Span,
    confidences: Sequence[float] | None,
) -> float | None:
    """计算片段内词元分数均值；该值尚未校准。"""
    if confidences is None:
        return None
    selected = confidences[span.start : span.end]
    if not all(0.0 <= value <= 1.0 for value in selected):
        raise SchemaValidationError("BIO置信度必须在0到1之间。")
    return sum(selected) / len(selected)


def build_parse_result(
    raw_text: str,
    labels: Sequence[str],
    *,
    confidences: Sequence[float] | None = None,
    model_version: str | None = None,
    confidence_transform: Callable[[str, float], tuple[float, bool]] | None = None,
) -> ParseResult:
    """组合模型片段和确定性约束字段，生成可追溯的本地解析结果。"""
    spans = spans_from_bio(raw_text, labels, confidences)
    constraints = extract_constraints(raw_text)
    fields = constraints.fields
    grouped: dict[str, list[Span]] = defaultdict(list)
    for span in spans:
        grouped[span.label].append(span)

    titles = grouped.get("TITLE", [])
    aliases = grouped.get("TITLE_ALIAS", [])
    if titles:
        fields.title = titles[0].text
        fields.title_aliases = [item.text for item in (*titles[1:], *aliases)]
    elif aliases:
        fields.title_aliases = [item.text for item in aliases]

    evidence: dict[str, Evidence] = {}
    for label, matching_spans in grouped.items():
        confidence_values = [
            value
            for span in matching_spans
            if (value := _mean_confidence(span, confidences)) is not None
        ]
        raw_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else None
        calibrated_confidence, calibrated = (
            confidence_transform(label, raw_confidence)
            if raw_confidence is not None and confidence_transform is not None
            else (raw_confidence, False)
        )
        evidence[label.lower()] = Evidence(
            spans=tuple(matching_spans),
            source="model",
            version=model_version,
            confidence=calibrated_confidence,
            calibrated=calibrated,
        )

    for hint in constraints.hints:
        evidence.setdefault(
            hint.label.lower(),
            Evidence(
                spans=(
                    Span(
                        start=hint.source.start,
                        end=hint.source.end,
                        text=hint.raw,
                        label=hint.label,  # 约束层只产生固定标签。
                        source="rule",
                    ),
                ),
                source="rule",
                version="1.0",
            ),
        )

    warnings = list(constraints.warnings)
    status = "ok"
    if not fields.title:
        status = "partial"
        warnings.append("模型尚未识别到主标题，已仅返回可确定的格式字段。")
    return ParseResult(
        raw_text=raw_text,
        extracted=fields,
        evidence=evidence,
        model_version=model_version,
        status=status,
        warnings=warnings,
    )
