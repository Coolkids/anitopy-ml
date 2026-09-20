"""仅供小样本连通性训练使用的字符模型定义。"""

from __future__ import annotations

import torch


class CharacterTagger(torch.nn.Module):
    """用字符嵌入验证训练与离线推理链路的最小 BIO 分类器。"""

    def __init__(self, vocabulary_size: int, label_count: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, 64, padding_idx=0)
        self.classifier = torch.nn.Linear(64, label_count)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """返回每个字符位置的 BIO 类别分数。"""
        return self.classifier(self.embedding(input_ids))
