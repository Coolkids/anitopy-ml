"""数据契约的关键行为测试。"""

import json
import unittest
from pathlib import Path

from anitopy_ml.errors import InputValidationError, SchemaValidationError
from anitopy_ml.provenance import validate_training_lineage
from anitopy_ml.schemas import (
    AnnotationMetadata,
    AnnotationRecord,
    BIO_LABELS,
    DataLineage,
    ExtractedFields,
    Evidence,
    ModelManifest,
    ParseResult,
    Span,
)


class SchemaTests(unittest.TestCase):
    """验证片段偏移和输入约束。"""

    def test_valid_evidence_round_trip(self) -> None:
        result = ParseResult(
            raw_text="示例作品 S01E03",
            evidence={
                "title": Evidence(
                    spans=(Span(0, 4, "示例作品", "TITLE"),),
                    source="human",
                )
            },
        )
        dumped = result.model_dump()
        self.assertEqual(dumped["evidence"]["title"]["spans"][0]["text"], "示例作品")

    def test_mismatched_span_is_rejected(self) -> None:
        result = ParseResult(
            raw_text="示例作品",
            evidence={"title": Evidence(spans=(Span(0, 2, "错误", "TITLE"),))},
        )
        with self.assertRaises(SchemaValidationError):
            result.validate()

    def test_empty_title_is_rejected(self) -> None:
        with self.assertRaises(InputValidationError):
            ParseResult(raw_text="  ").validate()

    def test_annotation_rejects_overlapping_spans(self) -> None:
        record = AnnotationRecord(
            sample_id="样本-1",
            raw_text="示例作品",
            spans=(Span(0, 4, "示例作品", "TITLE"), Span(1, 3, "例作", "TITLE_ALIAS")),
            extracted=ExtractedFields(title="示例作品"),
            annotation=AnnotationMetadata("gold", "accepted"),
            lineage=DataLineage(("user_csv", "human"), True),
        )
        with self.assertRaises(SchemaValidationError):
            record.validate()

    def test_model_manifest_and_label_file_are_stable(self) -> None:
        labels = tuple(json.loads(Path("configs/标签定义.json").read_text(encoding="utf-8"))["labels"])
        self.assertEqual(labels, BIO_LABELS)
        manifest = ModelManifest(
            model_name="示例模型",
            model_version="1.0",
            model_revision="abc",
            tokenizer_revision="abc",
            schema_version="1.0",
            normalizer_version="1.0",
            labels=labels,
            training_manifest_sha256="a" * 64,
        )
        self.assertEqual(manifest.label_to_id()["B-TITLE"], 1)

    def test_training_lineage_rejects_unapproved_data(self) -> None:
        with self.assertRaises(SchemaValidationError):
            validate_training_lineage([DataLineage(("tmdb",), False, "许可-待确认")])
