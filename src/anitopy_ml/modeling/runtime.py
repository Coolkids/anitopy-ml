"""可选训练依赖的显式检查。"""

from __future__ import annotations

import importlib
from typing import Any

from anitopy_ml.errors import ConfigurationError


def require_training_dependencies() -> tuple[Any, Any]:
    """加载 torch 和 transformers；缺失时给出不含网络操作的中文提示。"""
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ModuleNotFoundError as error:
        raise ConfigurationError(
            "未安装训练组件，请执行“uv sync --extra train”后再创建模型。"
        ) from error
    return torch, transformers
