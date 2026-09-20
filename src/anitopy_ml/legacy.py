"""冻结旧版 Anitopy 的显式加载与兼容调用。

该模块不把旧项目的绝对路径写入发布包。运行基线时，调用方必须显式提供
包含 ``anitopy`` 目录和 MPL-2.0 许可证文件的旧项目根目录。
"""

from __future__ import annotations

import importlib
import runpy
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from anitopy_ml.errors import ConfigurationError

LegacyParse = Callable[[str, dict[str, Any]], dict[str, Any] | None]


@dataclass(frozen=True, slots=True)
class FixtureMismatch:
    """旧用例的预期结果与实际结果的单条差异。"""

    index: int
    title: str
    expected: dict[str, Any]
    actual: dict[str, Any] | None


def _fixture_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    """将旧夹具中的 option_ 前缀转换为旧解析器选项。"""
    return {
        key.removeprefix("option_"): value
        for key, value in (options or {}).items()
    }


def _load_fixture_entries(root: Path, fixture: str) -> list[list[Any]]:
    """不导入当前项目 tests 包，直接读取指定旧夹具文件。"""
    fixture_paths = {
        "common": root / "tests" / "fixtures" / "table.py",
        "failing": root / "tests" / "fixtures" / "failing_table.py",
    }
    if fixture not in fixture_paths:
        raise ConfigurationError("夹具类型只能是 common 或 failing。")
    path = fixture_paths[fixture]
    if not path.is_file():
        raise ConfigurationError("旧解析器目录中未找到指定测试夹具。")
    namespace = runpy.run_path(str(path))
    variable_name = "table" if fixture == "common" else "failing_table"
    entries = namespace.get(variable_name)
    if not isinstance(entries, list):
        raise ConfigurationError("旧测试夹具结构无效。")
    return entries


def compare_legacy_fixture(source: str | Path, fixture: str) -> dict[str, Any]:
    """执行指定旧夹具并返回完整且可序列化的比较报告。"""
    root = _validate_source(source)
    legacy_parse = load_legacy_parse(root)
    entries = _load_fixture_entries(root, fixture)
    mismatches: list[FixtureMismatch] = []
    exceptions: list[dict[str, str | int]] = []

    for index, entry in enumerate(entries):
        title, options, expected = entry
        expected_result = {key: value for key, value in dict(expected).items() if key != "id"}
        try:
            actual = legacy_parse(title, options=_fixture_options(options))
        except Exception as error:  # 旧实现异常也属于基线结果
            exceptions.append({"index": index, "title": title, "error": repr(error)})
            continue
        if actual != expected_result:
            mismatches.append(
                FixtureMismatch(index=index, title=title, expected=expected_result, actual=actual)
            )

    return {
        "fixture": fixture,
        "source": str(root),
        "total": len(entries),
        "exact_match": len(entries) - len(mismatches) - len(exceptions),
        "mismatch_count": len(mismatches),
        "exception_count": len(exceptions),
        "mismatches": [asdict(item) for item in mismatches],
        "exceptions": exceptions,
    }


def compare_legacy_baseline(source: str | Path) -> dict[str, Any]:
    """执行常规和已知失败夹具，形成统一旧行为基线报告。"""
    reports = [
        compare_legacy_fixture(source, fixture="common"),
        compare_legacy_fixture(source, fixture="failing"),
    ]
    return {
        "source": reports[0]["source"],
        "fixtures": reports,
        "total": sum(int(item["total"]) for item in reports),
        "exact_match": sum(int(item["exact_match"]) for item in reports),
        "exception_count": sum(int(item["exception_count"]) for item in reports),
    }


def _validate_source(source: str | Path) -> Path:
    """验证旧项目根目录包含可加载代码和许可证。"""
    root = Path(source).expanduser().resolve()
    if not (root / "anitopy" / "__init__.py").is_file():
        raise ConfigurationError("旧解析器目录中未找到 anitopy 包。")
    if not (root / "LICENSE").is_file():
        raise ConfigurationError("旧解析器目录中未找到许可证文件。")
    return root


def load_legacy_parse(source: str | Path) -> LegacyParse:
    """从指定旧项目目录加载 ``parse``，且不污染当前项目导入路径。"""
    root = _validate_source(source)
    original_modules: dict[str, ModuleType] = {
        name: module
        for name, module in sys.modules.items()
        if name == "anitopy" or name.startswith("anitopy.")
    }
    for name in original_modules:
        sys.modules.pop(name, None)

    root_text = str(root)
    sys.path.insert(0, root_text)
    try:
        package = importlib.import_module("anitopy")
        parse = package.parse
    except ImportError as error:
        raise ConfigurationError("无法加载指定目录中的旧解析器。") from error
    finally:
        sys.path.remove(root_text)
        for name in tuple(sys.modules):
            if name == "anitopy" or name.startswith("anitopy."):
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)

    return parse


def parse_legacy(
    title: str,
    *,
    source: str | Path,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """以旧实现解析标题，并复制选项以避免旧实现修改调用方字典。"""
    if not isinstance(title, str):
        raise ConfigurationError("旧解析器的标题输入必须是字符串。")
    copied_options = dict(options or {})
    legacy_parse = load_legacy_parse(source)
    return legacy_parse(title, options=copied_options)
