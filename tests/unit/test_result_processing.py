"""验证结果集处理目前只保序透传。"""

from __future__ import annotations

import unittest

from anitopy_ml.processing.results import NoOpResultProcessor, ResultProcessingContext


class ResultProcessingTests(unittest.TestCase):
    """确认结果集预留入口不修改结果对象或顺序。"""

    def test_processor_returns_same_items_and_order(self) -> None:
        values = [object(), {"status": "error"}, object()]
        context = ResultProcessingContext("batch", "fingerprint")
        output = NoOpResultProcessor().process(values, context)
        self.assertIs(output, values)
        self.assertEqual(output, values)
