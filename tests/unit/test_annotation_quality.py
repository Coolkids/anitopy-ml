"""标注质量报告测试。"""

import unittest

from anitopy_ml.annotation.quality import build_quality_report, quality_markdown
from anitopy_ml.annotation.review import review_sample
from anitopy_ml.annotation.seed import build_seed_record
from anitopy_ml.annotation.store import AnnotationStore


class AnnotationQualityTests(unittest.TestCase):
    """确保报告只按最新版本计算训练资格。"""

    def test_report_counts_accepted_trainable_records(self) -> None:
        with AnnotationStore(":memory:") as store:
            store.save(build_seed_record("样本1", "示例作品 EP01"), actor="规则")
            review_sample(store, "样本1", actor="审核员", decision="approve")
            report = build_quality_report(store.latest_payloads())
        self.assertEqual(report["最新样本数"], 1)
        self.assertEqual(report["可训练已接受样本数"], 1)
        self.assertEqual(report["距首批300条差额"], 299)
        self.assertIn("EPISODE_EXPR", quality_markdown(report))
