"""媒体标题解析 HTTP 接口。"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_GET, require_POST

from anitopy_ml.errors import AnitopyMlError, ConfigurationError, InputValidationError
from anitopy_ml.webapi.service import (
    configured_device,
    configured_model_directory,
    get_parser,
    maximum_batch_size,
    maximum_title_length,
)


def _response(payload: dict[str, Any], *, status: int = 200) -> JsonResponse:
    """返回 UTF-8 JSON，确保中文提示不会被转义。"""
    return JsonResponse(payload, status=status, json_dumps_params={"ensure_ascii": False})


def _error(message: str, *, code: str, status: int) -> JsonResponse:
    """返回稳定的中文错误结构。"""
    return _response({"error": {"code": code, "message": message}}, status=status)


def _read_json(request: HttpRequest) -> dict[str, Any]:
    """读取并验证 JSON 对象请求体。"""
    content_type = request.headers.get("Content-Type", "")
    if not content_type.lower().startswith("application/json"):
        raise InputValidationError("请求Content-Type必须为application/json。")
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InputValidationError("请求体必须是UTF-8编码的合法JSON。") from error
    if not isinstance(payload, dict):
        raise InputValidationError("请求体必须是JSON对象。")
    return payload


def _title(value: object) -> str:
    """校验单条标题长度和类型。"""
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError("title必须是非空字符串。")
    if len(value) > maximum_title_length():
        raise InputValidationError("title超过当前服务允许的最大字符数。")
    return value


def _handle(handler: Callable[[], JsonResponse]) -> JsonResponse:
    """将项目异常映射为稳定 HTTP 响应。"""
    try:
        return handler()
    except InputValidationError as error:
        return _error(str(error), code=error.code, status=400)
    except ConfigurationError as error:
        return _error(str(error), code=error.code, status=503)
    except AnitopyMlError as error:
        return _error(str(error), code=error.code, status=422)


@require_GET
def health_view(request: HttpRequest) -> JsonResponse:
    """加载模型后返回就绪状态，供容器健康检查使用。"""
    del request

    def handle() -> JsonResponse:
        parser = get_parser()
        return _response(
            {
                "status": "ok",
                "model_directory": str(configured_model_directory()),
                "model_version": parser.model_version,
                "device": configured_device(),
            }
        )

    return _handle(handle)


@require_POST
def parse_view(request: HttpRequest) -> JsonResponse:
    """解析单条媒体标题。"""

    def handle() -> JsonResponse:
        payload = _read_json(request)
        title = _title(payload.get("title"))
        result = get_parser().parse(title)
        return _response({"result": result.model_dump()})

    return _handle(handle)


@require_POST
def parse_batch_view(request: HttpRequest) -> JsonResponse:
    """按输入顺序解析多条媒体标题。"""

    def handle() -> JsonResponse:
        payload = _read_json(request)
        titles = payload.get("titles")
        on_error = payload.get("on_error", "record")
        if not isinstance(titles, list) or not titles:
            raise InputValidationError("titles必须是非空数组。")
        if len(titles) > maximum_batch_size():
            raise InputValidationError("titles超过当前服务允许的批量上限。")
        if on_error not in {"record", "raise"}:
            raise InputValidationError("on_error只能是record或raise。")
        for title in titles:
            if isinstance(title, str) and len(title) > maximum_title_length():
                raise InputValidationError("titles中存在超过最大字符数的标题。")
        results = get_parser().parse_batch(titles, on_error=on_error)
        serialized = [item.model_dump() if hasattr(item, "model_dump") else item for item in results]
        return _response({"results": serialized})

    return _handle(handle)
