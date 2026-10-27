"""保守作品分组与稳定数据划分。"""

from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from anitopy_ml.data.ingest import TitleRecord
from anitopy_ml.normalizer import normalize_title

_METADATA = re.compile(
    r"(?i)\b(?:s\d{1,2}e\d+(?:\.\d+)?|ep?\d+(?:\.\d+)?|第\s*\d+\s*[集话話季]|"
    r"\d{1,4}p|4k|8k|x26[45]|h\.?26[45]|hevc|avc|aac|flac|"
    r"bdremux|bluray|web[- ]?dl|webrip|hdtv|内封|外挂|简繁|mkv|mp4)\b"
)
_SEPARATORS = re.compile(r"[\[\](){}._,;:|/\\\-—–]+")
_WHITESPACE = re.compile(r"\s+")
_LEADING_GROUP = re.compile(r"^\s*[\[【(（][^\]】)）]{1,40}[\]】)）]\s*")


@dataclass(frozen=True, slots=True)
class GroupAssignment:
    """一个样本的作品组与分区归属。"""

    sample_id: str
    work_group_id: str
    split: str
    reason: str
    grouping_key: str


def conservative_grouping_key(raw_text: str) -> str:
    """移除明确技术和季集元数据，保留标题主体；空键不自动合并。"""
    text = normalize_title(raw_text).model_text.casefold()
    text = _METADATA.sub(" ", text)
    text = _SEPARATORS.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def review_candidate_key(raw_text: str) -> str:
    """生成更宽松的复核键，只用于发现候选，绝不自动合并。"""
    text = normalize_title(raw_text).model_text
    text = _LEADING_GROUP.sub("", text)
    return conservative_grouping_key(text)


def _stable_split(group_id: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if value < 80 else "validation" if value < 90 else "test"


def build_group_assignments(
    records: Iterable[TitleRecord],
    *,
    seed: int = 20260917,
    overrides: dict[str, str] | None = None,
) -> tuple[list[GroupAssignment], dict[str, object]]:
    """使用人工覆盖或相同保守键分组，并按组确定性分区。"""
    overrides = overrides or {}
    grouped: dict[str, list[tuple[TitleRecord, str, str]]] = defaultdict(list)
    for record in records:
        override = overrides.get(record.sample_id)
        key = conservative_grouping_key(record.raw_text)
        if override:
            group_id, reason = override, "manual_override"
        elif key:
            group_id = "auto-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
            reason = "conservative_key"
        else:
            group_id, reason = "solo-" + record.sample_id, "empty_key"
        grouped[group_id].append((record, key, reason))

    assignments = [
        GroupAssignment(record.sample_id, group_id, _stable_split(group_id, seed), reason, key)
        for group_id, items in sorted(grouped.items())
        for record, key, reason in items
    ]
    counts = Counter(item.split for item in assignments)
    report = {
        "版本": 1,
        "随机种子": seed,
        "样本数": len(assignments),
        "作品组数": len(grouped),
        "分区样本数": dict(counts),
        "人工覆盖数": sum(item.reason == "manual_override" for item in assignments),
        "自动合并组数": sum(len(items) > 1 for items in grouped.values()),
    }
    audit_assignments(assignments)
    return assignments, report


def audit_assignments(assignments: Iterable[GroupAssignment]) -> None:
    """拒绝同一作品组跨分区，供训练导出前复用。"""
    splits: dict[str, set[str]] = defaultdict(set)
    for item in assignments:
        splits[item.work_group_id].add(item.split)
    leaked = [group_id for group_id, values in splits.items() if len(values) > 1]
    if leaked:
        raise ValueError(f"检测到作品组跨分区泄漏：{leaked[0]}")


def assignments_as_jsonl(assignments: Iterable[GroupAssignment]) -> list[dict[str, str]]:
    """转换为可写入JSONL的稳定结构。"""
    return [asdict(item) for item in assignments]


def write_split_outputs(
    assignments: Iterable[GroupAssignment],
    report: dict[str, object],
    *,
    output: str | Path,
    report_output: str | Path,
    source_sha256: str,
) -> None:
    """写入分区JSONL和可复核报告。"""
    items = assignments_as_jsonl(assignments)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
    complete_report = dict(report)
    complete_report["源文件SHA256"] = source_sha256
    complete_report["分区文件"] = str(target)
    report_target = Path(report_output)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(
        json.dumps(complete_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def find_review_candidates(
    records: Iterable[TitleRecord],
    assignments: Iterable[GroupAssignment],
    *,
    max_cluster_size: int = 30,
    similarity_threshold: float = 0.98,
    max_similarity_candidates: int = 500,
) -> list[dict[str, object]]:
    """找出宽松键相同或高度相似、但仍属不同组的人工复核候选。"""
    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("相似度阈值必须在0到1之间。")
    if max_similarity_candidates < 0:
        raise ValueError("相似候选数量上限不能小于0。")
    assignment_by_id = {item.sample_id: item for item in assignments}
    buckets: dict[str, list[TitleRecord]] = defaultdict(list)
    for record in records:
        key = review_candidate_key(record.raw_text)
        if len(key) >= 4:
            buckets[key].append(record)
    candidates: list[dict[str, object]] = []
    exact_candidate_keys: set[str] = set()
    for key, items in sorted(buckets.items()):
        group_ids = {assignment_by_id[item.sample_id].work_group_id for item in items}
        if len(group_ids) < 2 or len(items) > max_cluster_size:
            continue
        exact_candidate_keys.add(key)
        candidates.append(
            {
                "候选键": key,
                "原因": "去除前导方括号组名后的保守键相同",
                "作品组ID": sorted(group_ids),
                "样本": [
                    {
                        "sample_id": item.sample_id,
                        "raw_text": item.raw_text,
                        "current_group_id": assignment_by_id[item.sample_id].work_group_id,
                    }
                    for item in items
                ],
            }
        )
    similarity_buckets: dict[str, list[tuple[TitleRecord, str]]] = defaultdict(list)
    for record in records:
        key = review_candidate_key(record.raw_text)
        compact = re.sub(r"\s+", "", key)
        if len(compact) >= 6:
            similarity_buckets[compact[:5]].append((record, key))
    for _, items in sorted(similarity_buckets.items()):
        if sum("相似度" in item for item in candidates) >= max_similarity_candidates:
            break
        if len(items) > max_cluster_size:
            continue
        for index, (left, left_key) in enumerate(items):
            for right, right_key in items[index + 1 :]:
                if (
                    left_key == right_key
                    or left_key in exact_candidate_keys
                    or right_key in exact_candidate_keys
                ):
                    continue
                left_assignment = assignment_by_id[left.sample_id]
                right_assignment = assignment_by_id[right.sample_id]
                if left_assignment.work_group_id == right_assignment.work_group_id:
                    continue
                score = SequenceMatcher(None, left_key, right_key, autojunk=False).ratio()
                if score < similarity_threshold:
                    continue
                candidates.append(
                    {
                        "候选键": [left_key, right_key],
                        "原因": f"保守键相似度达到{similarity_threshold:.2f}，需人工确认",
                        "相似度": round(score, 4),
                        "作品组ID": sorted(
                            {left_assignment.work_group_id, right_assignment.work_group_id}
                        ),
                        "样本": [
                            {
                                "sample_id": left.sample_id,
                                "raw_text": left.raw_text,
                                "current_group_id": left_assignment.work_group_id,
                            },
                            {
                                "sample_id": right.sample_id,
                                "raw_text": right.raw_text,
                                "current_group_id": right_assignment.work_group_id,
                            },
                        ],
                    }
                )
                if sum("相似度" in item for item in candidates) >= max_similarity_candidates:
                    break
            if sum("相似度" in item for item in candidates) >= max_similarity_candidates:
                break
    return candidates


def write_review_candidates(
    candidates: Iterable[dict[str, object]],
    *,
    output: str | Path,
    report_output: str | Path,
) -> int:
    """写出疑似同组复核队列及其数量报告。"""
    items = list(candidates)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
    report_target = Path(report_output)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(
        json.dumps({"版本": 1, "疑似同组簇数": len(items), "队列文件": str(target)}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return len(items)
