"""数据来源与训练用途的统一检查。"""

from collections.abc import Iterable

from anitopy_ml.errors import SchemaValidationError
from anitopy_ml.schemas import DataLineage


def validate_training_lineage(lineages: Iterable[DataLineage]) -> int:
    """验证一组样本都允许训练，并返回已验证的数量。"""
    count = 0
    for count, lineage in enumerate(lineages, start=1):
        try:
            lineage.validate_for_training()
        except SchemaValidationError as error:
            raise SchemaValidationError(f"第{count}条样本的来源校验失败：{error}") from error
    return count
