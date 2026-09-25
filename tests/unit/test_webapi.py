"""Django 本地 API 的请求与错误边界测试。"""

from __future__ import annotations

import importlib.util
import json
import os
import unittest
from unittest.mock import patch

from anitopy_ml.errors import ConfigurationError

DJANGO_AVAILABLE = importlib.util.find_spec("django") is not None

if DJANGO_AVAILABLE:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "anitopy_ml.webapi.settings")
    import django

    django.setup()
    from django.test import Client


class _FakeResult:
    """用于隔离 HTTP 层的最小解析结果。"""

    def __init__(self, title: str) -> None:
        self.title = title

    def model_dump(self) -> dict[str, object]:
        return {"raw_text": self.title, "status": "ok", "warnings": []}


class _FakeParser:
    """不加载实际模型的解析器替身。"""

    model_version = "测试模型"

    def parse(self, title: str) -> _FakeResult:
        return _FakeResult(title)

    def parse_batch(self, titles: list[str], *, on_error: str) -> list[_FakeResult | dict[str, object]]:
        results: list[_FakeResult | dict[str, object]] = []
        for title in titles:
            if not title and on_error == "record":
                results.append({"raw_text": title, "status": "error", "warnings": ["标题不能为空。"]})
            else:
                results.append(_FakeResult(title))
        return results


@unittest.skipUnless(DJANGO_AVAILABLE, "未安装Django服务依赖。")
class WebApiTests(unittest.TestCase):
    """验证 JSON 契约不依赖实际模型权重。"""

    def setUp(self) -> None:
        self.client = Client()

    def test_parse_returns_serialized_result(self) -> None:
        with patch("anitopy_ml.webapi.views.get_parser", return_value=_FakeParser()):
            response = self.client.post(
                "/v1/parse",
                data=json.dumps({"title": "示例作品 - 01"}),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result"]["raw_text"], "示例作品 - 01")

    def test_batch_keeps_error_record_order(self) -> None:
        with patch("anitopy_ml.webapi.views.get_parser", return_value=_FakeParser()):
            response = self.client.post(
                "/v1/parse-batch",
                data=json.dumps({"titles": ["示例作品", ""], "on_error": "record"}),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["results"][1]["status"], "error")

    def test_invalid_content_type_returns_chinese_json_error(self) -> None:
        response = self.client.post("/v1/parse", data="{}", content_type="text/plain")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "INPUT_INVALID")

    def test_health_returns_service_unavailable_when_model_cannot_load(self) -> None:
        with patch(
            "anitopy_ml.webapi.views.get_parser",
            side_effect=ConfigurationError("模型目录不存在。"),
        ):
            response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "CONFIG_INVALID")
