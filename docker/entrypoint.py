#!/usr/bin/env python3
"""容器启动时准备持久化模型和基础编码器缓存。"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

_REQUIRED_MODEL_PATHS = (
    "best_model.pt",
    "checkpoint_metadata.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)
_SOURCE_RECORD = ".anitopy-model-source.json"
_UPDATE_MODES = {"missing", "always", "url_changed", "never"}


def _fail(message: str) -> None:
    """向容器日志写入明确的启动失败原因。"""
    print(f"模型准备失败：{message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def _is_complete_model(directory: Path) -> bool:
    """判断目录是否具备加载发布模型所需的最小文件集。"""
    return directory.is_dir() and all((directory / relative).is_file() for relative in _REQUIRED_MODEL_PATHS)


def _read_source_url(directory: Path) -> str | None:
    """读取上次成功写入该目录的模型来源地址。"""
    try:
        payload = json.loads((directory / _SOURCE_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    value = payload.get("model_url") if isinstance(payload, dict) else None
    return value if isinstance(value, str) else None


def _download(url: str, target: Path) -> None:
    """以有限重试下载模型包，避免依赖额外的系统工具。"""
    request = urllib.request.Request(url, headers={"User-Agent": "anitopy-ml-container"})
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as output:
                shutil.copyfileobj(response, output)
            return
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
            target.unlink(missing_ok=True)
            if attempt == 3:
                _fail(f"无法下载模型包：{error}")
            print(f"模型下载第{attempt}次失败，将重试：{error}", file=sys.stderr, flush=True)
            time.sleep(attempt * 2)


def _extract_safely(archive: Path, directory: Path) -> None:
    """拒绝包含目录穿越路径的 ZIP，再解压模型文件。"""
    try:
        with zipfile.ZipFile(archive) as package:
            destination = directory.resolve()
            for member in package.infolist():
                candidate = (directory / member.filename).resolve()
                if candidate != destination and destination not in candidate.parents:
                    _fail("模型 ZIP 包含非法文件路径。")
            package.extractall(directory)
    except zipfile.BadZipFile as error:
        _fail(f"模型包不是有效 ZIP 文件：{error}")


def _replace_model(directory: Path, url: str) -> None:
    """在同一文件系统中完成下载、校验和目录切换。"""
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.download-", dir=directory.parent))
    archive = temporary / "model.zip"
    try:
        print(f"正在下载模型包：{url}", flush=True)
        _download(url, archive)
        extracted = temporary / "model"
        extracted.mkdir()
        _extract_safely(archive, extracted)
        archive.unlink(missing_ok=True)
        if not _is_complete_model(extracted):
            _fail("模型包缺少权重、元数据或分词器文件。")
        (extracted / _SOURCE_RECORD).write_text(
            json.dumps({"model_url": url, "updated_at": int(time.time())}, ensure_ascii=False),
            encoding="utf-8",
        )
        backup = directory.with_name(f".{directory.name}.previous")
        shutil.rmtree(backup, ignore_errors=True)
        if directory.exists():
            directory.rename(backup)
        try:
            extracted.rename(directory)
        except OSError:
            if backup.exists() and not directory.exists():
                backup.rename(directory)
            raise
        shutil.rmtree(backup, ignore_errors=True)
        print(f"模型已写入持久化目录：{directory}", flush=True)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _prepare_model() -> None:
    """依据更新策略确保模型目录可供 API 离线加载。"""
    directory = Path(os.environ["ANITOPY_MODEL_DIR"])
    url = os.environ.get("ANITOPY_MODEL_URL", "").strip()
    mode = os.environ.get("ANITOPY_MODEL_UPDATE", "missing").lower()
    if mode not in _UPDATE_MODES:
        _fail("ANITOPY_MODEL_UPDATE 只能设置为 missing、always、url_changed 或 never。")
    complete = _is_complete_model(directory)
    should_download = (
        mode == "always"
        or (mode == "missing" and not complete)
        or (mode == "url_changed" and (not complete or _read_source_url(directory) != url))
    )
    if should_download:
        if not url:
            _fail("需要下载模型时，ANITOPY_MODEL_URL 不能为空。")
        _replace_model(directory, url)
    elif not complete:
        _fail(f"模型目录不完整：{directory}。请挂载完整模型卷或允许下载模型。")
    else:
        print(f"复用持久化模型：{directory}", flush=True)


def _prepare_base_model() -> None:
    """把基础编码器写入持久化缓存，随后强制 API 离线加载。"""
    base_model = os.environ.get("ANITOPY_BASE_MODEL", "").strip()
    if not base_model:
        _fail("ANITOPY_BASE_MODEL 不能为空。")
    try:
        from transformers import AutoModel, AutoTokenizer

        print(f"正在检查基础编码器缓存：{base_model}", flush=True)
        try:
            AutoTokenizer.from_pretrained(base_model, local_files_only=True, use_fast=True)
            AutoModel.from_pretrained(base_model, local_files_only=True)
        except OSError:
            print(f"基础编码器缓存缺失，正在下载：{base_model}", flush=True)
            AutoTokenizer.from_pretrained(base_model, use_fast=True)
            AutoModel.from_pretrained(base_model)
    except (OSError, ValueError, RuntimeError) as error:
        _fail(f"无法准备基础编码器缓存：{error}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main() -> None:
    """准备运行依赖后，将控制权交给镜像默认命令。"""
    if len(sys.argv) < 2:
        _fail("容器未收到需要执行的服务命令。")
    _prepare_model()
    _prepare_base_model()
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
