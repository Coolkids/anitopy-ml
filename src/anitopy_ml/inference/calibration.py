"""字段置信度的离线分桶校准与模型绑定校验。"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from anitopy_ml.errors import ConfigurationError, InputValidationError, SchemaValidationError


CalibrationObservation = tuple[str, float, bool]


class ConfidenceCalibrator:
    """仅为验证样本充足的字段返回经验校准置信度。"""

    def __init__(self, payload: Mapping[str, object]) -> None:
        fields = payload.get("字段")
        parameters = payload.get("参数")
        if not isinstance(fields, dict) or not isinstance(parameters, dict):
            raise SchemaValidationError("校准文件缺少字段或参数。")
        bucket_count = parameters.get("分桶数")
        if not isinstance(bucket_count, int) or bucket_count < 2:
            raise SchemaValidationError("校准文件的分桶数量无效。")
        self._payload = dict(payload)
        self._fields = fields
        self._bucket_count = bucket_count

    @property
    def payload(self) -> dict[str, object]:
        """返回可序列化的校准内容副本。"""
        return dict(self._payload)

    def calibrate(self, label: str, confidence: float) -> tuple[float, bool]:
        """返回字段的校准分数；未校准字段保留原始分数并明确标记。"""
        if not 0.0 <= confidence <= 1.0:
            raise InputValidationError("待校准置信度必须在0到1之间。")
        field = self._fields.get(label.lower())
        if not isinstance(field, dict):
            return confidence, False
        bins = field.get("分桶")
        if not isinstance(bins, list):
            return confidence, False
        index = min(int(confidence * self._bucket_count), self._bucket_count - 1)
        if index >= len(bins) or not isinstance(bins[index], dict):
            return confidence, False
        calibrated = bins[index].get("校准置信度")
        if not isinstance(calibrated, (int, float)) or not 0.0 <= float(calibrated) <= 1.0:
            return confidence, False
        return float(calibrated), True


class AcceptancePolicy:
    """将已校准字段分为可自动接收、需复核和未校准三种状态。"""

    def __init__(self, payload: Mapping[str, object]) -> None:
        thresholds = payload.get("字段自动接收阈值")
        legacy_threshold = payload.get("自动接收阈值")
        if thresholds is None and isinstance(legacy_threshold, (int, float)) and 0.0 <= float(legacy_threshold) <= 1.0:
            thresholds = {}
        if not isinstance(thresholds, dict) or not all(
            isinstance(label, str) and isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
            for label, value in thresholds.items()
        ):
            raise SchemaValidationError("接受策略的字段自动接收阈值无效。")
        self.thresholds = {label.lower(): float(value) for label, value in thresholds.items()}
        self.default_threshold = float(legacy_threshold) if not self.thresholds and isinstance(legacy_threshold, (int, float)) else None

    def decide(self, label: str, confidence: float | None, calibrated: bool) -> str:
        """基于已校准置信度返回可解释的处理建议。"""
        if confidence is None or not calibrated:
            return "未校准"
        threshold = self.thresholds.get(label.lower(), self.default_threshold)
        return "自动接收" if threshold is not None and confidence >= threshold else "需要复核"


def model_sha256(model_directory: str | Path) -> str:
    """计算候选检查点权重哈希，防止校准器被误用于其他模型。"""
    path = Path(model_directory) / "best_model.pt"
    if not path.is_file():
        raise InputValidationError(f"模型目录缺少best_model.pt：{path.parent}。")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_confidence_calibrator(
    observations: Iterable[CalibrationObservation],
    *,
    model_directory: str | Path,
    bucket_count: int = 10,
    minimum_samples: int = 20,
) -> dict[str, object]:
    """从验证集预测拟合字段经验置信度，样本不足字段不写入校准器。"""
    if bucket_count < 2 or minimum_samples < 1:
        raise InputValidationError("分桶数至少为2，每字段最小样本数至少为1。")
    grouped: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for label, confidence, correct in observations:
        if not label or not 0.0 <= confidence <= 1.0:
            raise SchemaValidationError("校准观测包含无效字段或置信度。")
        grouped[label.lower()].append((confidence, bool(correct)))
    fields: dict[str, object] = {}
    uncalibrated: dict[str, int] = {}
    for label, values in sorted(grouped.items()):
        if len(values) < minimum_samples:
            uncalibrated[label] = len(values)
            continue
        bins: list[dict[str, object]] = []
        for index in range(bucket_count):
            lower = index / bucket_count
            upper = (index + 1) / bucket_count
            selected = [correct for score, correct in values if min(int(score * bucket_count), bucket_count - 1) == index]
            correct_count = sum(selected)
            bins.append(
                {
                    "下界": lower,
                    "上界": upper,
                    "样本数": len(selected),
                    "正确数": correct_count,
                    # Beta(1, 1)平滑可防止小分桶出现虚假的0或1概率。
                    "校准置信度": (correct_count + 1) / (len(selected) + 2),
                }
            )
        fields[label] = {
            "样本数": len(values),
            "正确数": sum(correct for _, correct in values),
            "分桶": bins,
        }
    return {
        "版本": 1,
        "说明": "仅由模板分布内验证集拟合；未列出的字段没有校准，不代表真实发布标题正确率。",
        "模型": {"检查点SHA256": model_sha256(model_directory)},
        "参数": {"分桶数": bucket_count, "每字段最小样本数": minimum_samples},
        "字段": fields,
        "未校准字段": uncalibrated,
    }


def coverage_error_curve(
    observations: Iterable[CalibrationObservation], calibrator: ConfidenceCalibrator
) -> list[dict[str, float | int]]:
    """给出各阈值下已校准预测的覆盖率和经验错误率。"""
    calibrated = [
        (score, correct)
        for label, confidence, correct in observations
        if (score := calibrator.calibrate(label, confidence))[1]
    ]
    values = [(float(score), correct) for (score, _), correct in calibrated]
    result: list[dict[str, float | int]] = []
    for threshold in (index / 10 for index in range(0, 11)):
        selected = [correct for score, correct in values if score >= threshold]
        result.append(
            {
                "阈值": threshold,
                "已校准预测数": len(selected),
                "覆盖率": len(selected) / len(values) if values else 0.0,
                "经验错误率": 1 - sum(selected) / len(selected) if selected else 0.0,
            }
        )
    return result


def write_calibration(payload: Mapping[str, object], path: str | Path) -> Path:
    """写入与权重绑定的校准文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def load_calibrator(path: str | Path, *, model_directory: str | Path) -> ConfidenceCalibrator:
    """读取校准器并验证其检查点哈希与待加载模型完全一致。"""
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"无法读取校准文件：{target}。") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("模型"), dict):
        raise ConfigurationError("校准文件结构无效。")
    expected = payload["模型"].get("检查点SHA256")
    if expected != model_sha256(model_directory):
        raise ConfigurationError("校准文件与当前模型检查点不匹配。")
    return ConfidenceCalibrator(payload)


def derive_acceptance_policy(
    calibration_payload: Mapping[str, object], *, maximum_error_rate: float = 0.05
) -> dict[str, object]:
    """从验证集曲线选择满足目标错误率的最低阈值，保留最大覆盖率。"""
    if not 0.0 <= maximum_error_rate < 1.0:
        raise InputValidationError("目标错误率必须在0到1之间。")
    fields = calibration_payload.get("字段")
    model = calibration_payload.get("模型")
    if not isinstance(fields, dict) or not isinstance(model, dict):
        raise SchemaValidationError("校准文件缺少字段或模型信息。")
    thresholds: dict[str, float] = {}
    review_fields: list[str] = []
    accepted_total = accepted_correct = calibrated_total = 0
    for label, field in fields.items():
        if not isinstance(label, str) or not isinstance(field, dict) or not isinstance(field.get("分桶"), list):
            raise SchemaValidationError("校准字段的分桶结构无效。")
        bins = [item for item in field["分桶"] if isinstance(item, dict)]
        calibrated_total += sum(int(item.get("样本数", 0)) for item in bins)
        candidates = [
            item
            for item in bins
            if int(item.get("样本数", 0)) > 0
            and isinstance(item.get("校准置信度"), (int, float))
            and 1 - float(item["校准置信度"]) <= maximum_error_rate + 1e-12
        ]
        if not candidates:
            review_fields.append(label)
            continue
        threshold = min(float(item["校准置信度"]) for item in candidates)
        thresholds[label] = threshold
        for item in bins:
            confidence = item.get("校准置信度")
            if isinstance(confidence, (int, float)) and float(confidence) >= threshold:
                accepted_total += int(item.get("样本数", 0))
                accepted_correct += int(item.get("正确数", 0))
    if not thresholds:
        raise SchemaValidationError("没有字段达到目标错误率，不能生成自动接收策略。")
    return {
        "版本": 1,
        "说明": "阈值按字段根据模板分布内验证集确定；未校准、未达标字段或低于字段阈值的结果保留证据并提示复核。",
        "模型": dict(model),
        "字段自动接收阈值": thresholds,
        "需要复核字段": review_fields,
        "目标经验错误率": maximum_error_rate,
        "验证集覆盖率": accepted_total / calibrated_total if calibrated_total else 0.0,
        "验证集经验错误率": 1 - accepted_correct / accepted_total if accepted_total else 0.0,
        "已校准预测数": accepted_total,
    }


def write_acceptance_policy(payload: Mapping[str, object], path: str | Path) -> Path:
    """写入与校准器同目录的可审查接受策略。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def load_acceptance_policy(path: str | Path, *, model_directory: str | Path) -> AcceptancePolicy:
    """读取策略并验证它绑定的权重哈希。"""
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"无法读取接受策略文件：{target}。") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("模型"), dict):
        raise ConfigurationError("接受策略文件结构无效。")
    if payload["模型"].get("检查点SHA256") != model_sha256(model_directory):
        raise ConfigurationError("接受策略文件与当前模型检查点不匹配。")
    return AcceptancePolicy(payload)
