"""使用验证分区拟合并写入模型置信度校准文件。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from anitopy_ml.api import MediaParser
from anitopy_ml.errors import InputValidationError, SchemaValidationError
from anitopy_ml.inference.calibration import (
    ConfidenceCalibrator,
    coverage_error_curve,
    fit_confidence_calibrator,
    write_calibration,
)
from anitopy_ml.inference.decoding import spans_from_bio


def _read_validation_records(path: str | Path) -> list[dict[str, Any]]:
    """读取字符级BIO验证记录。"""
    target = Path(path)
    try:
        records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise InputValidationError(f"无法读取验证数据：{target}。") from error
    if not records:
        raise InputValidationError("验证数据为空，不能拟合校准器。")
    return records


def calibrate_extractor(
    *,
    model_directory: str | Path,
    validation_path: str | Path,
    output_path: str | Path,
    device: str = "auto",
    bucket_count: int = 10,
    minimum_samples: int = 20,
) -> dict[str, object]:
    """基于验证集模型预测拟合字段置信度；冻结测试集不会被读取。"""
    parser = MediaParser.from_pretrained(model_directory, device=device, use_calibration=False)
    observations: list[tuple[str, float, bool]] = []
    expected_field_count: dict[str, int] = defaultdict(int)
    for record in _read_validation_records(validation_path):
        text = record.get("text")
        labels = record.get("labels")
        if not isinstance(text, str) or not isinstance(labels, list) or len(text) != len(labels):
            raise SchemaValidationError("验证记录缺少与原文等长的字符级BIO标签。")
        expected: dict[str, tuple[tuple[int, int, str], ...]] = defaultdict(tuple)
        grouped: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        for span in spans_from_bio(text, labels):
            grouped[span.label.lower()].append((span.start, span.end, span.text))
        for label, spans in grouped.items():
            expected[label] = tuple(spans)
            expected_field_count[label] += 1
        result = parser.parse(text)
        for label, evidence in result.evidence.items():
            if evidence.source != "model" or evidence.confidence is None:
                continue
            predicted = tuple((span.start, span.end, span.text) for span in evidence.spans)
            observations.append((label, evidence.confidence, predicted == expected[label]))
    payload = fit_confidence_calibrator(
        observations,
        model_directory=model_directory,
        bucket_count=bucket_count,
        minimum_samples=minimum_samples,
    )
    calibrator = ConfidenceCalibrator(payload)
    payload["验证统计"] = {
        "验证样本数": len(_read_validation_records(validation_path)),
        "模型字段预测数": len(observations),
        "参考字段出现数": dict(sorted(expected_field_count.items())),
    }
    payload["覆盖率_错误率曲线"] = coverage_error_curve(observations, calibrator)
    write_calibration(payload, output_path)
    return payload
