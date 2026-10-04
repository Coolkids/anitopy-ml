"""将处理后标题上的模型证据投影回原始标题。"""

from __future__ import annotations

from dataclasses import replace

from anitopy_ml.normalizer import SourceRange
from anitopy_ml.processing.types import PreprocessingResult, ProcessingMetadata
from anitopy_ml.schemas import ParseResult, Span


def _merge_ranges(ranges: list[SourceRange]) -> tuple[SourceRange, ...]:
    if not ranges:
        return ()
    ordered = sorted(set(ranges), key=lambda item: (item.start, item.end))
    merged = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.start <= previous.end:
            merged[-1] = SourceRange(previous.start, max(previous.end, current.end))
        else:
            merged.append(current)
    return tuple(merged)


def project_result(result: ParseResult, processed: PreprocessingResult) -> ParseResult:
    """恢复原始输入及证据偏移，并附加预处理摘要。"""
    raw_text = processed.raw_text
    projected_evidence = {}
    for key, evidence in result.evidence.items():
        projected: list[Span] = []
        for span in evidence.spans:
            origins = [
                source
                for mapping in processed.mappings[span.start : span.end]
                for source in mapping
            ]
            for region in _merge_ranges(origins):
                if region.start == region.end:
                    continue
                projected.append(
                    Span(
                        start=region.start,
                        end=region.end,
                        text=raw_text[region.start : region.end],
                        label=span.label,
                        source=span.source,
                    )
                )
        projected_evidence[key] = replace(
            evidence,
            spans=tuple(projected),
            calibrated=(False if processed.changed and evidence.source == "model" else evidence.calibrated),
        )

    result.raw_text = raw_text
    result.evidence = projected_evidence
    result.schema_version = "1.1"
    result.preprocessing = ProcessingMetadata(
        config_version=processed.config_version,
        config_fingerprint=processed.config_fingerprint,
        processed_text=processed.processed_text,
        changed=processed.changed,
        matched_rules=processed.matched_rules,
    )
    return result
