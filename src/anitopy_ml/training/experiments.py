"""汇总同一训练配置下的重复实验，避免用单次结果选择候选模型。"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Sequence

from anitopy_ml.errors import InputValidationError, SchemaValidationError


def _read_report(directory: str | Path) -> dict[str, Any]:
    """读取一个训练目录中的报告，并检查稳定性比较所需字段。"""
    path = Path(directory) / "训练报告.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InputValidationError(f"无法读取训练报告：{path}。") from error
    required = (
        "最佳验证实体F1",
        "耗时秒",
        "随机种子",
        "训练数据清单SHA256",
        "训练样本数",
        "验证样本数",
        "训练配置",
    )
    if not isinstance(payload, dict) or any(name not in payload for name in required):
        raise SchemaValidationError(f"训练报告缺少稳定性比较所需字段：{path}。")
    return payload


def summarize_repeated_runs(run_directories: Sequence[str | Path]) -> dict[str, object]:
    """汇总重复训练的验证结果，并拒绝混入不同数据或训练配置的实验。"""
    if not run_directories:
        raise InputValidationError("至少需要提供一个训练目录。")
    reports = [_read_report(directory) for directory in run_directories]
    data_hashes = {str(report["训练数据清单SHA256"]) for report in reports}
    configs = {json.dumps(report["训练配置"], ensure_ascii=False, sort_keys=True) for report in reports}
    sample_sizes = {
        (int(report["训练样本数"]), int(report["验证样本数"]))
        for report in reports
    }
    if len(data_hashes) != 1 or len(configs) != 1 or len(sample_sizes) != 1:
        raise SchemaValidationError("重复实验的数据清单、样本数量或训练配置不一致，不能汇总。")

    seeds = [int(report["随机种子"]) for report in reports]
    if len(set(seeds)) != len(seeds):
        raise SchemaValidationError("重复实验包含相同随机种子，不能作为独立重复实验。")
    rows = [
        {
            "训练目录": str(Path(directory)),
            "随机种子": int(report["随机种子"]),
            "最佳验证实体F1": float(report["最佳验证实体F1"]),
            "耗时秒": float(report["耗时秒"]),
            "训练轮次": int(report.get("训练轮次", 0)),
            "设备": str(report.get("设备", "未知")),
        }
        for directory, report in zip(run_directories, reports, strict=True)
    ]
    scores = [float(row["最佳验证实体F1"]) for row in rows]
    durations = [float(row["耗时秒"]) for row in rows]
    best = max(rows, key=lambda row: float(row["最佳验证实体F1"]))
    return {
        "说明": "结果只适用于模板分布内验证，不代表真实发布标题准确率。",
        "实验数量": len(rows),
        "是否满足三随机种子要求": len(rows) >= 3,
        "训练数据清单SHA256": next(iter(data_hashes)),
        "训练样本数": next(iter(sample_sizes))[0],
        "验证样本数": next(iter(sample_sizes))[1],
        "训练配置": reports[0]["训练配置"],
        "验证实体F1": {
            "平均值": statistics.fmean(scores),
            "标准差": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "最小值": min(scores),
            "最大值": max(scores),
        },
        "耗时秒": {
            "平均值": statistics.fmean(durations),
            "总计": sum(durations),
        },
        "最佳候选": best,
        "实验": rows,
    }


def write_repeated_runs_summary(
    run_directories: Sequence[str | Path], output_path: str | Path
) -> dict[str, object]:
    """写入可审查的重复实验稳定性报告。"""
    summary = summarize_repeated_runs(run_directories)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
