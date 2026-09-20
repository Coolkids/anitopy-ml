"""使用确定性规则生成待审核的弱标注建议。"""

from __future__ import annotations

import json
from pathlib import Path

from anitopy_ml.annotation.store import AnnotationStore
from anitopy_ml.constraints import extract_constraints
from anitopy_ml.schemas import AnnotationMetadata, AnnotationRecord, DataLineage, Span


def build_seed_record(sample_id: str, raw_text: str) -> AnnotationRecord:
    """从明确格式生成弱标注；不猜测标题实体，也不授予训练许可。"""
    constraints = extract_constraints(raw_text)
    spans = tuple(
        Span(
            start=hint.source.start,
            end=hint.source.end,
            text=hint.raw,
            label=hint.label,  # 片段类型已由约束层限制为固定标签。
            source="rule",
        )
        for hint in constraints.hints
    )
    return AnnotationRecord(
        sample_id=sample_id,
        raw_text=raw_text,
        spans=spans,
        extracted=constraints.fields,
        annotation=AnnotationMetadata(tier="weak", review_status="pending"),
        lineage=DataLineage(sources=("user_csv", "rule"), training_allowed=False),
    )


def seed_from_jsonl(
    input_path: str | Path, database: str | Path, *, limit: int | None = None
) -> int:
    """读取导入后的JSONL，将规则建议以弱标注写入SQLite。"""
    processed = 0
    created = 0
    with AnnotationStore(database) as store, Path(input_path).open(encoding="utf-8") as stream:
        for line in stream:
            if limit is not None and processed >= limit:
                break
            processed += 1
            row = json.loads(line)
            if store.latest_payload(row["sample_id"]) is not None:
                continue
            record = build_seed_record(row["sample_id"], row["raw_text"])
            store.save(record, actor="规则预标注", action="生成弱标注建议")
            created += 1
    return created
