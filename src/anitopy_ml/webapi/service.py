"""进程内模型加载和配置校验。"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from anitopy_ml.api import MediaParser
from anitopy_ml.errors import ConfigurationError

_PARSER: MediaParser | None = None
_LOCK = threading.Lock()


def configured_model_directory() -> Path:
    """读取模型目录环境变量并返回规范路径。"""
    return Path(os.environ.get("ANITOPY_MODEL_DIR", "artifacts/releases/anitopy-ml-v11"))


def configured_device() -> str:
    """读取并校验推理设备配置。"""
    device = os.environ.get("ANITOPY_DEVICE", "auto").lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("ANITOPY_DEVICE 只能设置为auto、cpu或cuda。")
    return device


def maximum_batch_size() -> int:
    """读取批量请求上限，避免单个请求耗尽本地资源。"""
    raw_value = os.environ.get("ANITOPY_MAX_BATCH_SIZE", "100")
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ConfigurationError("ANITOPY_MAX_BATCH_SIZE 必须是正整数。") from error
    if not 1 <= value <= 1_000:
        raise ConfigurationError("ANITOPY_MAX_BATCH_SIZE 必须在1到1000之间。")
    return value


def maximum_title_length() -> int:
    """读取单条标题字符数上限。"""
    raw_value = os.environ.get("ANITOPY_MAX_TITLE_LENGTH", "4096")
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ConfigurationError("ANITOPY_MAX_TITLE_LENGTH 必须是正整数。") from error
    if not 1 <= value <= 16_384:
        raise ConfigurationError("ANITOPY_MAX_TITLE_LENGTH 必须在1到16384之间。")
    return value


def get_parser() -> MediaParser:
    """懒加载并复用当前进程的离线模型实例。"""
    global _PARSER
    if _PARSER is not None:
        return _PARSER
    with _LOCK:
        if _PARSER is None:
            directory = configured_model_directory()
            if not directory.is_dir():
                raise ConfigurationError(f"模型目录不存在：{directory}。")
            _PARSER = MediaParser.from_pretrained(directory, device=configured_device())
    return _PARSER


def reset_parser_for_testing() -> None:
    """清空进程缓存，仅供自动化测试使用。"""
    global _PARSER
    with _LOCK:
        _PARSER = None
