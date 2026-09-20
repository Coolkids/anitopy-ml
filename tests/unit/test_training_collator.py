"""动态填充与对齐问题输出测试。"""

import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from anitopy_ml.training.collator import dynamic_pad
from anitopy_ml.training.issues import write_alignment_issues


class _NonClosingStringIO(io.StringIO):
    """允许测试在上下文退出后读取写入内容。"""

    def close(self) -> None:
        """保留内容供断言使用。"""


class TrainingCollatorTests(unittest.TestCase):
    """验证填充值和问题记录不参与损失。"""

    def test_dynamic_pad_uses_ignore_label_and_zero_masks(self) -> None:
        batch = [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [3, 4], "loss_mask": [True, True]},
            {"input_ids": [5], "attention_mask": [1], "labels": [6], "loss_mask": [True]},
        ]
        result = dynamic_pad(batch, pad_token_id=0)
        self.assertEqual(result["input_ids"][1], [5, 0])
        self.assertEqual(result["labels"][1], [6, -100])
        self.assertEqual(result["loss_mask"][1], [1, 0])

    def test_alignment_issue_output_requires_reason(self) -> None:
        stream = _NonClosingStringIO()
        with patch.object(Path, "open", return_value=stream):
            count = write_alignment_issues([{"sample_id": "样本1", "reason": "跨越实体边界"}], "问题.jsonl")
        self.assertEqual(count, 1)
        self.assertEqual(json.loads(stream.getvalue())["reason"], "跨越实体边界")
