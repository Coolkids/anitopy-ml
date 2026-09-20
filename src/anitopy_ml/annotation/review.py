"""弱标注的人工审核、版本升级和审计操作。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

from anitopy_ml.annotation.store import AnnotationStore
from anitopy_ml.errors import ConfigurationError
from anitopy_ml.schemas import (
    AnnotationMetadata,
    AnnotationRecord,
    CatalogMatch,
    DataLineage,
    ExtractedFields,
    Span,
)


def record_from_payload(payload: dict[str, Any]) -> AnnotationRecord:
    """将SQLite中的JSON恢复为经过Schema校验的标注记录。"""
    catalog_data = payload.get("catalog")
    catalog = CatalogMatch(**catalog_data) if catalog_data else None
    annotation_data = dict(payload["annotation"])
    annotation_data["unlabeled_regions"] = tuple(
        tuple(region) for region in annotation_data.get("unlabeled_regions", [])
    )
    lineage_data = dict(payload["lineage"])
    lineage_data["sources"] = tuple(lineage_data["sources"])
    return AnnotationRecord(
        sample_id=payload["sample_id"],
        raw_text=payload["raw_text"],
        spans=tuple(Span(**span) for span in payload["spans"]),
        extracted=ExtractedFields(**payload["extracted"]),
        annotation=AnnotationMetadata(**annotation_data),
        lineage=DataLineage(**lineage_data),
        catalog=catalog,
        schema_version=payload.get("schema_version", "1.0"),
    )


def _next_lineage(record: AnnotationRecord, *, actor: str, training_allowed: bool) -> DataLineage:
    """追加人工审核来源，保留已有来源和授权引用。"""
    sources = tuple(dict.fromkeys((*record.lineage.sources, "human")))
    return DataLineage(
        sources=sources,
        training_allowed=training_allowed,
        authorization_ref=record.lineage.authorization_ref,
    )


def review_sample(
    store: AnnotationStore,
    sample_id: str,
    *,
    actor: str,
    decision: Literal["approve", "reject"],
    tier: Literal["gold", "silver"] = "gold",
) -> AnnotationRecord:
    """保存人工审核决定，并使用下一版本避免覆盖并发修改。"""
    if not actor.strip():
        raise ConfigurationError("审核人员不能为空。")
    payload = store.latest_payload(sample_id)
    if payload is None:
        raise ConfigurationError("未找到需要审核的样本。")
    record = record_from_payload(payload)
    if decision == "approve":
        metadata = AnnotationMetadata(
            tier=tier,
            review_status="accepted",
            reviewer_id=actor,
            version=record.annotation.version + 1,
            unlabeled_regions=record.annotation.unlabeled_regions,
        )
        updated = replace(
            record,
            annotation=metadata,
            lineage=_next_lineage(record, actor=actor, training_allowed=True),
        )
        store.save(updated, actor=actor, action="人工审核通过")
        return updated
    metadata = AnnotationMetadata(
        tier="review",
        review_status="rejected",
        reviewer_id=actor,
        version=record.annotation.version + 1,
        unlabeled_regions=record.annotation.unlabeled_regions,
    )
    updated = replace(
        record,
        annotation=metadata,
        lineage=_next_lineage(record, actor=actor, training_allowed=False),
    )
    store.save(updated, actor=actor, action="人工审核拒绝")
    return updated


def mark_unresolved(
    store: AnnotationStore,
    sample_id: str,
    *,
    actor: str,
) -> AnnotationRecord:
    """将无法确认的标题保存为待复核版本，且始终禁止用于训练。"""
    if not actor.strip():
        raise ConfigurationError("审核人员不能为空。")
    payload = store.latest_payload(sample_id)
    if payload is None:
        raise ConfigurationError("未找到需要审核的样本。")
    record = record_from_payload(payload)
    metadata = AnnotationMetadata(
        tier="review",
        review_status="needs_review",
        reviewer_id=actor,
        version=record.annotation.version + 1,
        unlabeled_regions=((0, len(record.raw_text)),),
    )
    updated = replace(
        record,
        spans=(),
        extracted=ExtractedFields(),
        annotation=metadata,
        lineage=_next_lineage(record, actor=actor, training_allowed=False),
    )
    store.save(updated, actor=actor, action="标记为无法判断")
    return updated


def save_manual_revision(
    store: AnnotationStore,
    sample_id: str,
    payload: dict[str, Any],
    *,
    actor: str,
) -> AnnotationRecord:
    """保存人工编辑后的待复核版本，保留原始样本ID和版本控制。"""
    if not actor.strip():
        raise ConfigurationError("审核人员不能为空。")
    current_payload = store.latest_payload(sample_id)
    if current_payload is None:
        raise ConfigurationError("未找到需要编辑的样本。")
    if payload.get("sample_id") != sample_id or payload.get("raw_text") != current_payload["raw_text"]:
        raise ConfigurationError("人工编辑不能修改样本ID或原始标题。")
    if payload.get("annotation", {}).get("version") != current_payload["annotation"]["version"]:
        raise ConfigurationError("编辑内容已过期，请重新加载后再保存。")
    record = record_from_payload(payload)
    metadata = AnnotationMetadata(
        tier="review",
        review_status="needs_review",
        reviewer_id=actor,
        version=record.annotation.version + 1,
        unlabeled_regions=record.annotation.unlabeled_regions,
    )
    updated = replace(
        record,
        annotation=metadata,
        lineage=_next_lineage(record, actor=actor, training_allowed=False),
    )
    store.save(updated, actor=actor, action="人工编辑待复核")
    return updated
