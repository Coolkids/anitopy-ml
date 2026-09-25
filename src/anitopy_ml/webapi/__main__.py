"""启动仅供本机开发使用的 Django API 服务。"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    """以 Django 开发服务器启动本地 API。"""
    parser = argparse.ArgumentParser(description="启动 Anitopy-ML 本地 Web API 服务。")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅允许本机访问。")
    parser.add_argument("--port", type=int, default=8000, help="监听端口，默认8000。")
    arguments = parser.parse_args()
    if not 1 <= arguments.port <= 65535:
        parser.error("监听端口必须在1到65535之间。")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "anitopy_ml.webapi.settings")
    from django.core.management import execute_from_command_line

    execute_from_command_line(
        ["anitopy-ml-webapi", "runserver", f"{arguments.host}:{arguments.port}", "--noreload"]
    )


if __name__ == "__main__":
    main()
