"""对已固定模型执行一次独立测试分区评测。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol, Sequence

from anitopy_ml.api import MediaParser
from anitopy_ml.errors import InputValidationError, SchemaValidationError
from anitopy_ml.inference.decoding import spans_from_bio
from anitopy_ml.schemas import ParseResult, Span


class ParserProtocol(Protocol):
    """使评测聚合逻辑可在不加载大模型时被测试。"""

    def parse(self, title: str) -> ParseResult: ...


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


def evaluate_records(records: Sequence[dict[str, Any]], parser: ParserProtocol) -> dict[str, object]:
    """聚合实体、字段和整条标题的严格边界指标。"""
    total_predicted = total_expected = total_correct = 0
    per_field: dict[str, dict[str, int]] = defaultdict(lambda: {"适用样本数": 0, "完全正确数": 0, "预测实体数": 0, "真实实体数": 0, "正确实体数": 0})
    full_correct = 0
    policy = {"自动接收": 0, "需要复核": 0, "未校准": 0}
    errors: list[dict[str, object]] = []
    for record in records:
        text = record.get("text")
        labels = record.get("labels")
        sample_id = record.get("sample_id", "未知样本")
        if not isinstance(text, str) or not isinstance(labels, list) or len(text) != len(labels):
            raise SchemaValidationError("冻结测试记录缺少与原文等长的字符级BIO标签。")
        expected_spans = spans_from_bio(text, labels)
        result = parser.parse(text)
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
