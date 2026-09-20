"""将已审核标注导出为可供训练对齐的字符级BIO中间数据。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from anitopy_ml.errors import SchemaValidationError
from anitopy_ml.schemas import BIO_LABELS


def read_split_assignments(path: str | Path) -> dict[str, str]:
    """读取冻结分区文件，返回样本ID到分区的映射。"""
    assignments: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                item = json.loads(line)
                sample_id, split = str(item["sample_id"]), str(item["split"])
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise SchemaValidationError(f"分区文件第{line_number}行无效。") from error
            if split not in {"train", "validation", "test"}:
                raise SchemaValidationError(f"分区文件第{line_number}行包含未知分区。")
            assignments[sample_id] = split
    return assignments


def read_work_group_ids(path: str | Path) -> dict[str, str]:
    """从冻结分区文件读取样本对应的作品组ID。"""
    values: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                item = json.loads(line)
                values[str(item["sample_id"])] = str(item["work_group_id"])
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise SchemaValidationError(f"分区文件第{line_number}行缺少作品组ID。") from error
    return values


def audit_effective_work_groups(
    assignments: dict[str, str],
    work_group_ids: dict[str, str],
) -> None:
    """拒绝人工合并后同一有效作品组跨越冻结分区。"""
    group_splits: dict[str, set[str]] = defaultdict(set)
    for sample_id, group_id in work_group_ids.items():
        split = assignments.get(sample_id)
        if split is not None and group_id:
            group_splits[group_id].add(split)
    leaked = sorted(group_id for group_id, splits in group_splits.items() if len(splits) > 1)
    if leaked:
        raise SchemaValidationError(
            f"人工作品组合并后跨越冻结分区：{leaked[0]}。请重新执行data split后再导出训练数据。"
        )


def payload_to_bio(
    payload: dict[str, Any],
    split: str,
    work_group_id: str | None = None,
) -> dict[str, Any]:
    """将单条已允许训练的最新标注转换为字符级BIO标签。"""
    annotation = payload["annotation"]
    lineage = payload["lineage"]
    raw_text = str(payload["raw_text"])
    if annotation["review_status"] != "accepted" or not lineage["training_allowed"]:
        raise SchemaValidationError("只能导出已接受且允许训练的标注。")
    if annotation.get("unlabeled_regions"):
        raise SchemaValidationError("含未标注区域的样本不能推断为O标签。")
    labels = ["O"] * len(raw_text)
    for span in sorted(payload.get("spans", []), key=lambda item: (item["start"], item["end"])):
        label = str(span["label"])
        start, end = int(span["start"]), int(span["end"])
        if not 0 <= start < end <= len(raw_text) or raw_text[start:end] != span["text"]:
            raise SchemaValidationError("训练导出的片段与原始标题不一致。")
        if any(item != "O" for item in labels[start:end]):
            raise SchemaValidationError("训练导出的片段不能重叠。")
        labels[start] = f"B-{label}"
        labels[start + 1 : end] = [f"I-{label}"] * (end - start - 1)
    if any(label not in BIO_LABELS for label in labels):
        raise SchemaValidationError("训练导出包含未知BIO标签。")
    result = {
        "sample_id": payload["sample_id"],
        "split": split,
        "text": raw_text,
        "characters": list(raw_text),
        "labels": labels,
        "schema_version": payload.get("schema_version", "1.0"),
    }
    if work_group_id:
        result["work_group_id"] = work_group_id
    return result


def build_training_splits(
    payloads: Iterable[dict[str, Any]],
    assignments: dict[str, str],
    *,
    work_group_ids: dict[str, str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """按冻结分区导出合格样本，并统计排除原因。"""
    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: Counter[str] = Counter()
    for payload in payloads:
        sample_id = str(payload["sample_id"])
        if sample_id not in assignments:
            skipped["缺少冻结分区"] += 1
            continue
        try:
            item = payload_to_bio(
                payload,
                assignments[sample_id],
                (work_group_ids or {}).get(sample_id),
            )
        except SchemaValidationError as error:
            skipped[str(error)] += 1
            continue
        splits[item["split"]].append(item)
    return {name: splits.get(name, []) for name in ("train", "validation", "test")}, dict(skipped)


def write_training_splits(
    splits: dict[str, list[dict[str, Any]]],
    skipped: dict[str, int],
    *,
    output_dir: str | Path,
) -> dict[str, object]:
    """写入三个BIO JSONL文件及含哈希的训练数据清单。"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, object]] = {}
    for split, items in splits.items():
        target = directory / f"{split}.jsonl"
        digest = hashlib.sha256()
        with target.open("w", encoding="utf-8", newline="\n") as stream:
            for item in items:
                line = json.dumps(item, ensure_ascii=False, sort_keys=True)
                stream.write(line + "\n")
                digest.update((line + "\n").encode("utf-8"))
        outputs[split] = {"文件": str(target), "样本数": len(items), "SHA256": digest.hexdigest()}
    manifest = {
        "版本": 1,
        "格式": "字符级BIO JSONL",
        "标签": list(BIO_LABELS),
        "分区": outputs,
        "排除统计": skipped,
    }
    (directory / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
