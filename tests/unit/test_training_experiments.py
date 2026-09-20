"""重复训练稳定性报告测试。"""

import unittest
from unittest.mock import patch

from anitopy_ml.errors import SchemaValidationError
from anitopy_ml.training.experiments import summarize_repeated_runs


def _report(*, seed: int, score: float, data_hash: str = "同一清单") -> dict[str, object]:
    return {
        "最佳验证实体F1": score,
        "耗时秒": 10.0,
        "随机种子": seed,
        "训练数据清单SHA256": data_hash,
        "训练样本数": 10,
        "验证样本数": 2,
        "训练轮次": 3,
        "训练配置": {"批大小": 2},
    }


class RepeatedRunSummaryTests(unittest.TestCase):
    """验证汇总只接受可比较且独立的重复实验。"""

    def test_summarizes_distinct_seeds(self) -> None:
        with patch(
            "anitopy_ml.training.experiments._read_report",
            side_effect=[_report(seed=1, score=0.90), _report(seed=2, score=0.93), _report(seed=3, score=0.92)],
        ):
            summary = summarize_repeated_runs(["一", "二", "三"])
        self.assertTrue(summary["是否满足三随机种子要求"])
        self.assertAlmostEqual(summary["验证实体F1"]["平均值"], 0.9166666666666666)
        self.assertEqual(summary["最佳候选"]["随机种子"], 2)

    def test_rejects_incomparable_manifests(self) -> None:
        with patch(
            "anitopy_ml.training.experiments._read_report",
            side_effect=[_report(seed=1, score=0.90), _report(seed=2, score=0.91, data_hash="另一清单")],
        ):
            with self.assertRaises(SchemaValidationError):
                summarize_repeated_runs(["一", "二"])
