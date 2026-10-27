"""独立于软件版本和编码器版本的模型发布信息与打包。"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Any

from anitopy_ml.errors import ConfigurationError, InputValidationError


RELEASE_FILENAME = "model_release.json"
REQUIRED_MODEL_PATHS = (
    "best_model.pt",
    "checkpoint_metadata.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+\-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def validate_model_version(value: object) -> str:
    """校验可用于发布标签的版本字符串，避免路径和空白混入版本号。"""
    if not isinstance(value, str) or not _VERSION.fullmatch(value):
        raise ConfigurationError("模型版本号须为1至128个英文字母、数字、点、下划线、加号或连字符，并以字母或数字开头。")
    return value


def read_model_release(directory: str | Path) -> dict[str, Any] | None:
    """读取模型包声明的实际版本；旧包缺少发布信息时保留兼容行为。"""
    target = Path(directory) / RELEASE_FILENAME
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError
        validate_model_version(payload.get("model_version"))
        files = payload["files"]
        if not isinstance(files, dict) or not set(REQUIRED_MODEL_PATHS) <= files.keys():
            raise ValueError
        for name, digest in files.items():
            relative = PurePosixPath(name)
            if (not name or "\\" in name or ":" in name or relative.is_absolute()
                    or ".." in relative.parts or str(relative) != name or name == RELEASE_FILENAME
                    or not isinstance(digest, str) or not _SHA256.fullmatch(digest)):
                raise ValueError
    except (OSError, TypeError, KeyError, ValueError) as error:
        raise ConfigurationError(f"模型发布信息无效：{target}。") from error
    return payload


def model_version(directory: str | Path) -> str:
    """优先返回发布版本，旧模型继续返回原来的目录标识。"""
    release = read_model_release(directory)
    return release["model_version"] if release is not None else f"XLM-R抽取模型：{Path(directory).name}"


def verify_model_release(directory: str | Path, *, expected_version: str = "") -> dict[str, Any] | None:
    """安装前校验目标版本和发布包内的逐文件哈希。"""
    root = Path(directory)
    if not all((root / name).is_file() for name in REQUIRED_MODEL_PATHS):
        raise ConfigurationError("模型包缺少权重、元数据或分词器文件。")
    release = read_model_release(root)
    if expected_version:
        validate_model_version(expected_version)
        if release is None or release["model_version"] != expected_version:
            actual = release["model_version"] if release is not None else "未声明版本"
            raise ConfigurationError(f"模型包版本与目标版本不一致：目标{expected_version}，实际{actual}。")
    if release is not None:
        for name, expected_hash in release["files"].items():
            path = root / name
            if root.resolve() not in path.resolve().parents or path.is_symlink() or not path.is_file():
                raise ConfigurationError(f"模型包文件缺失或路径无效：{name}。")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != expected_hash:
                raise ConfigurationError(f"模型包文件SHA256校验失败：{name}。")
    return release


def model_information(directory: str | Path) -> dict[str, Any]:
    """查询磁盘模型信息，不载入权重或联网。"""
    root = Path(directory)
    if not root.is_dir():
        raise InputValidationError(f"模型目录不存在：{root}。")
    release = read_model_release(root)
    return {
        "model_directory": str(root),
        "model_version": model_version(root),
        "version_source": RELEASE_FILENAME if release is not None else "directory_name",
        "complete": all((root / name).is_file() for name in REQUIRED_MODEL_PATHS),
        "weights_sha256": release["files"]["best_model.pt"] if release is not None else None,
    }


def package_model(directory: str | Path, *, version: str, output: str | Path) -> dict[str, Any]:
    """流式打包推理文件并生成版本清单，保留训练目录原样。"""
    validate_model_version(version)
    root = Path(directory)
    target = Path(output)
    if not all((root / name).is_file() for name in REQUIRED_MODEL_PATHS):
        raise InputValidationError("源模型目录缺少权重、元数据或分词器文件。")
    if target.exists():
        raise InputValidationError(f"模型包已存在，请使用新的输出路径：{target}。")
    if root.resolve() == target.resolve() or root.resolve() in target.resolve().parents:
        raise InputValidationError("模型包输出路径必须位于源模型目录之外。")
    files = [root / "best_model.pt", root / "checkpoint_metadata.json"]
    files.extend(sorted(path for path in (root / "tokenizer").rglob("*") if path.is_file()))
    files.extend(root / name for name in ("calibration.json", "acceptance_policy.json") if (root / name).is_file())
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = NamedTemporaryFile(prefix=target.name + ".", suffix=".tmp", dir=target.parent, delete=False)
    temporary = Path(temporary_handle.name)
    hashes: dict[str, str] = {}
    try:
        with temporary_handle as raw, zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in files:
                if root.resolve() not in path.resolve().parents or path.is_symlink():
                    raise InputValidationError("源模型目录不能包含指向外部的文件或符号链接。")
                name = path.relative_to(root).as_posix()
                digest = hashlib.sha256()
                with path.open("rb") as source, archive.open(name, "w", force_zip64=True) as destination:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(block)
                        digest.update(block)
                hashes[name] = digest.hexdigest()
            release = {"schema_version": 1, "model_version": version, "files": hashes}
            archive.writestr(RELEASE_FILENAME, json.dumps(release, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    with target.open("rb") as stream:
        archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"model_version": version, "file": str(target), "sha256": archive_hash, "files": len(hashes)}
