"""预留的结果集处理边界及本期恒等透传实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, Protocol, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ResultProcessingContext:
    """一次解析调用的处理模式和配置指纹。"""

    mode: Literal["single", "batch"]
    config_fingerprint: str


class ResultProcessor(Protocol[T]):
    """定义可替换的结果集处理入口。"""

    def process(self, items: Sequence[T], context: ResultProcessingContext) -> Sequence[T]:
        """接收并返回顺序、长度不变的解析结果集合。"""
        ...


class NoOpResultProcessor(Generic[T]):
    """本期只透传输入序列，不执行结果集业务逻辑。"""

    def process(self, items: Sequence[T], context: ResultProcessingContext) -> Sequence[T]:
        del context
        return items
