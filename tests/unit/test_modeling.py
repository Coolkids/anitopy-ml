"""BIO解码与实体评测测试。"""

import unittest

from anitopy_ml.modeling.decoder import constrained_bio_decode, transition_allowed
from anitopy_ml.modeling.metrics import entity_metrics


class ModelingTests(unittest.TestCase):
    """验证解码不会从I标签开头且指标要求边界完全一致。"""

    def test_decoder_avoids_illegal_initial_inside_label(self) -> None:
        labels = ["O", "B-TITLE", "I-TITLE"]
        decoded = constrained_bio_decode([[0.0, 1.0, 10.0], [0.0, 1.0, 2.0]], labels)
        self.assertEqual(decoded, ["B-TITLE", "I-TITLE"])
        self.assertFalse(transition_allowed(None, "I-TITLE"))

    def test_entity_metrics_require_exact_boundary(self) -> None:
        metrics = entity_metrics(
            ["B-TITLE", "I-TITLE", "O"],
            ["B-TITLE", "O", "O"],
        )
        self.assertEqual(metrics["正确实体数"], 0)
        self.assertEqual(metrics["F1"], 0.0)
