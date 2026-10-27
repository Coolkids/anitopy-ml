"""以固定内存生成、去重和划分大规模模板数据。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from anitopy_ml.data.synthetic import (
    _bio_labels,
    _work_key,
    _work_split_assignments,
    iter_synthetic_records,
    load_template_data,
    synthetic_template_entry_key,
)
from anitopy_ml.errors import InputValidationError, SchemaValidationError
from anitopy_ml.schemas import BIO_LABELS


def prepare_large_synthetic(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    count: int,
    seed: int = 20261025,
    split_seed: int = 20261025,
    optional_rate: float = 1.0,
    season_rate: float = 0.7,
) -> dict[str, object]:
    """逐条生成并用磁盘索引去重，全部审计通过后才发布分区清单。"""
    try:
        template_hash = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    except OSError as error:
        raise InputValidationError("无法读取模板数据JSON。") from error
    payload = load_template_data(input_path)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise InputValidationError("大数据输出目录必须为空，避免覆盖已冻结的数据版本。")
    output.mkdir(parents=True, exist_ok=True)
    assignments = _work_split_assignments({_work_key(item) for item in payload["title"]}, seed=split_seed)
    expected_entries = {
        f"synthetic-template-group-{group_index:02d}-entry-{entry_index:02d}"
        for group_index, group in enumerate(payload["name_template"], 1)
        for entry_index, _ in enumerate(group["name"], 1)
    }
    names = ("train", "validation", "test")
    coverage: dict[str, Counter[str]] = {name: Counter() for name in names}
    fields: dict[str, Counter[str]] = {name: Counter() for name in names}
    seen_works: set[str] = set()
    counts: Counter[str] = Counter()
    hashes = {name: hashlib.sha256() for name in names}
    duplicates = 0
    with TemporaryDirectory(prefix=".prepare-", dir=output) as temporary, ExitStack() as stack:
        staging = Path(temporary)
        connection = sqlite3.connect(staging / "dedupe.sqlite")
        stack.callback(connection.close)
        connection.execute("PRAGMA cache_size = -8192")
        connection.execute("CREATE TABLE seen (text TEXT PRIMARY KEY, labels TEXT NOT NULL, work_key TEXT NOT NULL)")
        streams = {name: stack.enter_context((staging / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n")) for name in names}
        for index, record in enumerate(iter_synthetic_records(payload, count=count, seed=seed, optional_rate=optional_rate, season_rate=season_rate), 1):
            text = str(record["text"])
            if record["characters"] != list(text) or record["labels"] != _bio_labels(text, record["spans"]):
                raise SchemaValidationError("生成样本的字符、BIO标签与实体片段不一致。")
            labels = json.dumps(record["labels"], ensure_ascii=False)
            work_key = str(record["work_key"])
            previous = connection.execute("SELECT labels, work_key FROM seen WHERE text = ?", (text,)).fetchone()
            if previous is not None:
                if previous != (labels, work_key):
                    raise SchemaValidationError("同一文本出现不同BIO标签或作品归属，无法自动去重。")
                duplicates += 1
                continue
            connection.execute("INSERT INTO seen VALUES (?, ?, ?)", (text, labels, work_key))
            if index % 10000 == 0:
                connection.commit()
            split = assignments[work_key]
            record["split"] = split
            record["work_group_id"] = "synthetic-work-" + hashlib.sha256(work_key.encode("utf-8")).hexdigest()[:16]
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            streams[split].write(line)
            hashes[split].update(line.encode("utf-8"))
            counts[split] += 1
            coverage[split][synthetic_template_entry_key(record)] += 1
            fields[split].update(str(span["label"]) for span in record["spans"])
            seen_works.add(work_key)
        missing = {name: sorted(expected_entries - coverage[name].keys()) for name in names if expected_entries - coverage[name].keys()}
        if missing or seen_works != assignments.keys():
            raise SchemaValidationError(f"数据覆盖不足，请增加生成数量或检查模板条件；缺失模板：{missing}；未覆盖作品数：{len(assignments.keys() - seen_works)}。")
        if hashlib.sha256(Path(input_path).read_bytes()).hexdigest() != template_hash:
            raise InputValidationError("生成期间模板文件发生变化，请固定模板后重新准备数据。")
        manifest = {
            "版本": 2,
            "格式": "模板合成字符级BIO JSONL",
            "标签": list(BIO_LABELS),
            "随机种子": seed,
            "分区随机种子": split_seed,
            "模板文件SHA256": template_hash,
            "生成参数": {"数量": count, "可选字段填充比例": optional_rate, "含季数标题比例": season_rate},
            "划分策略": "作品隔离且每个分区覆盖全部模板条目",
            "说明": "仅衡量模板分布内的泛化；模板结构在三个分区间共享。",
            "原始样本数": count,
            "重复文本数": duplicates,
            "BIO不一致样本数": 0,
            "作品条目数": len(seen_works),
            "模板条目数": len(expected_entries),
            "分区模板条目覆盖": {name: dict(sorted(coverage[name].items())) for name in names},
            "分区字段覆盖": {name: dict(sorted(fields[name].items())) for name in names},
            "作品分区": dict(sorted(assignments.items())),
            "分区": {name: {"文件": str(output / f"{name}.jsonl"), "样本数": counts[name], "SHA256": hashes[name].hexdigest()} for name in names},
        }
        (staging / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Windows 下必须先关闭文件句柄，再移动已完成的分区文件。
        stack.close()
        for name in names:
            (staging / f"{name}.jsonl").replace(output / f"{name}.jsonl")
        (staging / "training_manifest.json").replace(output / "training_manifest.json")
    return manifest
