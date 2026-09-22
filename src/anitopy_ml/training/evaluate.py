"""对已固定模型执行一次独立测试分区评测。"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol, Sequence

from anitopy_ml.api import MediaParser
from anitopy_ml.constraints import parse_season_expression
from anitopy_ml.errors import InputValidationError, SchemaValidationError
from anitopy_ml.inference.decoding import spans_from_bio
from anitopy_ml.schemas import ParseResult, Span


class ParserProtocol(Protocol):
    """使评测聚合逻辑可在不加载大模型时被测试。"""

    def parse(self, title: str) -> ParseResult: ...


_NAME_LABELS = frozenset({"TITLE", "TITLE_ALIAS"})
_EPISODE_LABELS = frozenset({"EPISODE_EXPR", "EPISODE_COUNT_EXPR"})


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    """读取冻结测试分区，并拒绝空文件或无效JSON。"""
    target = Path(path)
    try:
        records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise InputValidationError(f"无法读取冻结测试数据：{target}。") from error
    if not records:
        raise InputValidationError("冻结测试数据为空，无法评测。")
    return records


def _span_keys(spans: Sequence[Span]) -> set[tuple[str, int, int]]:
    """转换为严格实体比较所需的标签和原文边界键。"""
    return {(span.label, span.start, span.end) for span in spans}


def _combined_extent(
    keys: set[tuple[str, int, int]], labels: frozenset[str]
) -> tuple[int, int] | None:
    """取得一组核心字段的整体边界，允许名称与别名合并为一个实体。"""
    ranges = [(start, end) for label, start, end in keys if label in labels]
    if not ranges:
        return None
    return min(start for start, _ in ranges), max(end for _, end in ranges)


def _numeric_value(raw: str) -> str | None:
    """将正数文本标准化为可稳定比较的十进制字符串。"""
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return None
    if value <= 0:
        return None
    return format(value.normalize(), "f")


def _expected_seasons(spans: Sequence[Span]) -> tuple[int, ...]:
    """从金标季数片段构造整数季数集合。"""
    return tuple(sorted({value for span in spans if span.label == "SEASON_EXPR" if (value := parse_season_expression(span.text)) is not None}))


def _expected_episodes(spans: Sequence[Span]) -> tuple[tuple[str, ...], ...]:
    """从金标集数片段构造单集、范围和总集数的语义集合。"""
    values: set[tuple[str, ...]] = set()
    for span in spans:
        if span.label not in _EPISODE_LABELS:
            continue
        range_match = re.search(r"(?P<start>\d+(?:\.\d+)?)\s*(?:~|-|至|到)\s*(?P<end>\d+(?:\.\d+)?)", span.text)
        if span.label == "EPISODE_COUNT_EXPR" and range_match:
            start = _numeric_value(range_match.group("start"))
            end = _numeric_value(range_match.group("end"))
            if start and end:
                values.add(("范围", start, end))
                values.add(("总集数", end))
            continue
        number_match = re.search(r"\d+(?:\.\d+)?", span.text)
        if number_match and (value := _numeric_value(number_match.group(0))):
            kind = "总集数" if span.label == "EPISODE_COUNT_EXPR" else "单集"
            values.add((kind, value))
    return tuple(sorted(values))


def _predicted_episodes(extracted: object) -> tuple[tuple[str, ...], ...]:
    """从解析输出构造与金标一致的集数语义集合。"""
    values: set[tuple[str, ...]] = set()
    for item in getattr(extracted, "episodes", []):
        if isinstance(item, dict) and (value := _numeric_value(str(item.get("value", "")))):
            values.add(("单集", value))
    for item in getattr(extracted, "episode_ranges", []):
        if not isinstance(item, dict):
            continue
        start = _numeric_value(str(item.get("start", "")))
        end = _numeric_value(str(item.get("end", "")))
        if start and end:
            values.add(("范围", start, end))
    declared = getattr(extracted, "declared_episode_count", None)
    if isinstance(declared, int) and declared > 0:
        values.add(("总集数", str(declared)))
    return tuple(sorted(values))


def _core_checks(
    expected: set[tuple[str, int, int]], predicted: set[tuple[str, int, int]], expected_spans: Sequence[Span], extracted: object
) -> dict[str, bool | None]:
    """计算发布核心字段的样本级正确性，季集按结构化数值语义比较。"""
    expected_name = _combined_extent(expected, _NAME_LABELS)
    predicted_name = _combined_extent(predicted, _NAME_LABELS)
    expected_season = _expected_seasons(expected_spans)
    predicted_season = tuple(sorted(set(getattr(extracted, "seasons", []))))
    expected_episode = _expected_episodes(expected_spans)
    predicted_episode = _predicted_episodes(extracted)
    return {
        "名称与别名联合": expected_name == predicted_name if expected_name is not None else None,
        "季数": expected_season == predicted_season if expected_season else None,
        "集数": expected_episode == predicted_episode if expected_episode else None,
    }


def evaluate_records(records: Sequence[dict[str, Any]], parser: ParserProtocol) -> dict[str, object]:
    """聚合实体、字段和整条标题的严格边界指标。"""
    total_predicted = total_expected = total_correct = 0
    per_field: dict[str, dict[str, int]] = defaultdict(lambda: {"适用样本数": 0, "完全正确数": 0, "预测实体数": 0, "真实实体数": 0, "正确实体数": 0})
    full_correct = 0
    core_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"适用样本数": 0, "完全正确数": 0}
    )
    core_full_applicable = core_full_correct = 0
    policy = {"自动接收": 0, "需要复核": 0, "未校准": 0}
    errors: list[dict[str, object]] = []
    parsed_results: list[ParseResult] | None = None
    if isinstance(parser, MediaParser):
        parsed_results = []
        for offset in range(0, len(records), 32):
            titles = [str(record.get("text", "")) for record in records[offset : offset + 32]]
            batch = parser.parse_batch(titles, on_error="raise")
            if not all(isinstance(result, ParseResult) for result in batch):
                raise SchemaValidationError("模型批量评测返回了无效解析结果。")
            parsed_results.extend(batch)
    for record_index, record in enumerate(records):
        text = record.get("text")
        labels = record.get("labels")
        sample_id = record.get("sample_id", "未知样本")
        if not isinstance(text, str) or not isinstance(labels, list) or len(text) != len(labels):
            raise SchemaValidationError("冻结测试记录缺少与原文等长的字符级BIO标签。")
        expected_spans = spans_from_bio(text, labels)
        result = parsed_results[record_index] if parsed_results is not None else parser.parse(text)
        predicted_spans = tuple(
            span for evidence in result.evidence.values() if evidence.source == "model" for span in evidence.spans
        )
        expected_keys = _span_keys(expected_spans)
        predicted_keys = _span_keys(predicted_spans)
        correct_keys = expected_keys & predicted_keys
        total_expected += len(expected_keys)
        total_predicted += len(predicted_keys)
        total_correct += len(correct_keys)
        if predicted_keys == expected_keys:
            full_correct += 1
        checks = _core_checks(expected_keys, predicted_keys, expected_spans, result.extracted)
        applicable_checks = [
            (name, correct) for name, correct in checks.items() if correct is not None
        ]
        for name, correct in applicable_checks:
            core_stats[name]["适用样本数"] += 1
            core_stats[name]["完全正确数"] += int(correct)
        if applicable_checks:
            core_full_applicable += 1
            core_full_correct += int(all(correct for _, correct in applicable_checks))
        labels_in_sample = {label for label, _, _ in expected_keys | predicted_keys}
        for label in labels_in_sample:
            expected_field = {key for key in expected_keys if key[0] == label}
            predicted_field = {key for key in predicted_keys if key[0] == label}
            stats = per_field[label]
            stats["适用样本数"] += 1
            stats["预测实体数"] += len(predicted_field)
            stats["真实实体数"] += len(expected_field)
            stats["正确实体数"] += len(expected_field & predicted_field)
            stats["完全正确数"] += int(expected_field == predicted_field)
        for evidence in result.evidence.values():
            if evidence.source != "model":
                continue
            if hasattr(parser, "acceptance_decision"):
                decision = parser.acceptance_decision(
                    evidence.spans[0].label.lower() if evidence.spans else "",
                    evidence.confidence,
                    evidence.calibrated,
                )
            elif not evidence.calibrated:
                decision = "未校准"
            elif evidence.confidence is not None and evidence.confidence >= 0.7:
                decision = "自动接收"
            else:
                decision = "需要复核"
            policy[decision] += 1
        if predicted_keys != expected_keys and len(errors) < 100:
            errors.append(
                {
                    "样本ID": sample_id,
                    "标题": text,
                    "缺失实体": sorted(expected_keys - predicted_keys),
                    "多余实体": sorted(predicted_keys - expected_keys),
                    "警告": result.warnings,
                }
            )
    precision = total_correct / total_predicted if total_predicted else 0.0
    recall = total_correct / total_expected if total_expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    field_report = {
        label: {
            **stats,
            "完全正确率": stats["完全正确数"] / stats["适用样本数"] if stats["适用样本数"] else 0.0,
        }
        for label, stats in sorted(per_field.items())
    }
    core_report = {
        name: {
            **stats,
            "完全正确率": stats["完全正确数"] / stats["适用样本数"]
            if stats["适用样本数"]
            else 0.0,
        }
        for name, stats in sorted(core_stats.items())
    }
    return {
        "说明": "这是未参与训练、候选选择或校准的模板分布内冻结测试；不代表真实发布标题准确率。",
        "测试样本数": len(records),
        "实体指标": {
            "预测实体数": total_predicted,
            "真实实体数": total_expected,
            "正确实体数": total_correct,
            "精确率": precision,
            "召回率": recall,
            "F1": f1,
        },
        "整条严格完全正确数": full_correct,
        "整条严格完全正确率": full_correct / len(records),
        "核心字段指标": core_report,
        "核心字段整条完全正确数": core_full_correct,
        "核心字段整条完全正确率": core_full_correct / core_full_applicable if core_full_applicable else 0.0,
        "字段指标": field_report,
        "接受策略统计": policy,
        "误差样本": errors,
    }


def evaluate_extractor(
    *, model_directory: str | Path, test_path: str | Path, output_path: str | Path, device: str = "auto"
) -> dict[str, object]:
    """加载固定候选和校准器，对冻结测试分区写入可审查报告。"""
    records = _read_records(test_path)
    parser = MediaParser.from_pretrained(model_directory, device=device)
    report = evaluate_records(records, parser)
    report["模型目录"] = str(model_directory)
    report["冻结测试文件"] = str(test_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def evaluate_validation_extractor(
    *, model_directory: str | Path, validation_path: str | Path, output_path: str | Path, device: str = "auto"
) -> dict[str, object]:
    """只使用验证分区评测候选模型，绝不读取冻结测试。"""
    records = _read_records(validation_path)
    parser = MediaParser.from_pretrained(model_directory, device=device)
    report = evaluate_records(records, parser)
    report["说明"] = "这是候选选择用的模板验证集评测，不代表真实发布标题准确率，也不能替代冻结测试。"
    report["验证样本数"] = report.pop("测试样本数")
    report["模型目录"] = str(model_directory)
    report["验证文件"] = str(validation_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
