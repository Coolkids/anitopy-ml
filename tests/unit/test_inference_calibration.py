"""置信度校准器测试。"""

import unittest
from unittest.mock import patch

from anitopy_ml.inference.calibration import (
    AcceptancePolicy,
    ConfidenceCalibrator,
    derive_acceptance_policy,
    fit_confidence_calibrator,
)


class ConfidenceCalibrationTests(unittest.TestCase):
    """确保校准只覆盖样本充分的字段，且不伪造未知字段概率。"""

    @patch("anitopy_ml.inference.calibration.model_sha256", return_value="检查点哈希")
    def test_fits_sufficient_field_and_leaves_sparse_field_uncalibrated(self, _: object) -> None:
        payload = fit_confidence_calibrator(
            [("TITLE", 0.91, True), ("TITLE", 0.92, True), ("TITLE", 0.93, False), ("YEAR", 0.8, True)],
            model_directory="模型目录",
            bucket_count=2,
            minimum_samples=3,
        )
        calibrator = ConfidenceCalibrator(payload)
        score, calibrated = calibrator.calibrate("title", 0.92)
        self.assertTrue(calibrated)
        self.assertAlmostEqual(score, 0.6)
        self.assertEqual(calibrator.calibrate("year", 0.8), (0.8, False))

    def test_derives_lowest_threshold_that_meets_error_target(self) -> None:
        policy = derive_acceptance_policy(
            {
                "模型": {"检查点SHA256": "检查点哈希"},
                "字段": {
                    "title": {"分桶": [{"样本数": 20, "正确数": 18, "校准置信度": 0.86}]},
                    "episode_expr": {"分桶": [{"样本数": 20, "正确数": 20, "校准置信度": 0.95}]},
                },
            },
            maximum_error_rate=0.05,
        )
        self.assertEqual(policy["字段自动接收阈值"], {"episode_expr": 0.95})
        self.assertEqual(AcceptancePolicy(policy).decide("title", 0.99, True), "需要复核")
        self.assertEqual(AcceptancePolicy(policy).decide("episode_expr", 0.95, True), "自动接收")
