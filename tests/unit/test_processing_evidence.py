"""验证预处理后的证据区间仍指向输入原文。"""

from __future__ import annotations

import unittest

from anitopy_ml.processing.config import parse_processing_config
from anitopy_ml.processing.evidence import project_result
from anitopy_ml.processing.preprocessing import PreprocessingPipeline
from anitopy_ml.processing.types import ProcessingMetadata
from anitopy_ml.schemas import Evidence, ParseResult, Span


class ProcessingEvidenceTests(unittest.TestCase):
    """覆盖屏蔽间隙、变长替换和可选摘要序列化。"""

    def test_mask_keeps_deleted_region_out_of_evidence(self) -> None:
        config = parse_processing_config(
            {
                "config_version": 1,
                "preprocessing": {
                    "enabled": True,
                    "rules": [{"id": "mask", "type": "string_mask", "match": "[广告]"}],
                },
            }
        )
        processed = PreprocessingPipeline(config).apply("A[广告]B")
        result = ParseResult(
            raw_text=processed.processed_text,
            evidence={
                "title": Evidence(
                    spans=(Span(0, 2, "AB", "TITLE"),),
                    source="model",
                    confidence=0.99,
                    calibrated=True,
                )
            },
        )
        projected = project_result(result, processed)
        projected.validate()
        self.assertEqual(projected.raw_text, "A[广告]B")
        self.assertEqual(
            [(span.start, span.end, span.text) for span in projected.evidence["title"].spans],
            [(0, 1, "A"), (5, 6, "B")],
        )
        self.assertFalse(projected.evidence["title"].calibrated)
        self.assertEqual(projected.schema_version, "1.1")
        self.assertIsInstance(projected.preprocessing, ProcessingMetadata)

    def test_expanding_replacement_maps_all_output_to_valid_original_slice(self) -> None:
        config = parse_processing_config(
            {
                "config_version": 1,
                "preprocessing": {
                    "enabled": True,
                    "rules": [{"id": "expand", "type": "string_replace", "match": "🙂", "replacement": "emoji"}],
                },
            }
        )
        processed = PreprocessingPipeline(config).apply("作品🙂")
        result = ParseResult(
            raw_text=processed.processed_text,
            evidence={"title": Evidence(spans=(Span(2, 7, "emoji", "TITLE"),), source="model")},
        )
        projected = project_result(result, processed)
        projected.validate()
        self.assertEqual(projected.evidence["title"].spans[0].text, "🙂")
        self.assertEqual((projected.evidence["title"].spans[0].start, projected.evidence["title"].spans[0].end), (2, 3))

    def test_disabled_processing_field_is_absent_from_legacy_dump(self) -> None:
        payload = ParseResult(raw_text="标题").model_dump()
        self.assertNotIn("preprocessing", payload)
