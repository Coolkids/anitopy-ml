"""抽取模型的固定配置与标签契约。"""

from __future__ import annotations

from dataclasses import dataclass

from anitopy_ml.errors import SchemaValidationError
from anitopy_ml.schemas import BIO_LABELS


@dataclass(frozen=True, slots=True)
class ExtractorModelConfig:
    """模型、分词器和BIO标签的可复现配置。"""

    model_name: str = "FacebookAI/xlm-roberta-base"
    model_revision: str = "main"
    tokenizer_revision: str = "main"
    labels: tuple[str, ...] = BIO_LABELS
    dropout: float = 0.1

    def validate(self) -> None:
        """校验配置不悄悄改变标签表或概率参数。"""
        if not self.model_name.strip() or not self.model_revision.strip() or not self.tokenizer_revision.strip():
            raise SchemaValidationError("模型和分词器版本不能为空。")
        if self.labels != BIO_LABELS:
            raise SchemaValidationError("抽取模型标签必须与项目BIO标签表完全一致。")
        if not 0.0 <= self.dropout < 1.0:
            raise SchemaValidationError("模型dropout必须在0到1之间。")

    def label_to_id(self) -> dict[str, int]:
        """按项目固定顺序生成标签编号。"""
        self.validate()
        return {label: index for index, label in enumerate(self.labels)}
