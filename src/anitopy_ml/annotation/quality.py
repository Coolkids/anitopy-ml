"""人工标注的质量与覆盖度报告。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def build_quality_report(payloads: list[dict[str, Any]]) -> dict[str, object]:
    """统计最新版本的审核状态、训练资格和片段覆盖度。"""
    tiers = Counter(str(item["annotation"]["tier"]) for item in payloads)
    statuses = Counter(str(item["annotation"]["review_status"]) for item in payloads)
    labels = Counter(
        str(span["label"])
        for item in payloads
        for span in item.get("spans", [])
    )
    trainable = [
        item
        for item in payloads
        if item["annotation"]["review_status"] == "accepted"
        and bool(item["lineage"]["training_allowed"])
    ]
    return {
        "版本": 1,
        "最新样本数": len(payloads),
        "层级统计": dict(sorted(tiers.items())),
        "审核状态统计": dict(sorted(statuses.items())),
        "可训练已接受样本数": len(trainable),
        "距首批300条差额": max(0, 300 - len(trainable)),
        "片段标签统计": dict(sorted(labels.items())),
        "无片段样本数": sum(not item.get("spans") for item in payloads),
    }


def quality_markdown(report: dict[str, object]) -> str:
    """以中文Markdown输出适合人工复核的质量摘要。"""
    def rows(values: dict[str, object]) -> str:
        return "\n".join(f"| {name} | {count} |" for name, count in values.items()) or "| 无 | 0 |"

    return (
        "# 金标质量报告\n\n"
        f"- 最新样本数：{report['最新样本数']}\n"
        f"- 可训练已接受样本数：{report['可训练已接受样本数']}\n"
        f"- 距首批300条差额：{report['距首批300条差额']}\n"
        f"- 无片段样本数：{report['无片段样本数']}\n\n"
        "## 标注层级\n\n| 层级 | 数量 |\n| --- | --- |\n"
        + rows(report["层级统计"])
        + "\n\n## 审核状态\n\n| 状态 | 数量 |\n| --- | --- |\n"
        + rows(report["审核状态统计"])
        + "\n\n## 片段标签覆盖\n\n| 标签代码 | 数量 |\n| --- | --- |\n"
        + rows(report["片段标签统计"])
        + "\n"
    )


def write_quality_report(report: dict[str, object], output: str | Path) -> None:
    """写入UTF-8质量报告。"""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(quality_markdown(report), encoding="utf-8")
