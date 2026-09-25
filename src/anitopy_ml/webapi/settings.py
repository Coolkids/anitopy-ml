"""不依赖数据库的 Django API 服务配置。"""

from __future__ import annotations

import os

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "anitopy-ml-local-api-only")
DEBUG = False
ALLOWED_HOSTS = [
    item.strip()
    for item in os.environ.get("ANITOPY_ALLOWED_HOSTS", "*").split(",")
    if item.strip()
]
ROOT_URLCONF = "anitopy_ml.webapi.urls"
MIDDLEWARE: list[str] = []
INSTALLED_APPS: list[str] = []
TEMPLATES: list[dict[str, object]] = []
USE_TZ = True
DEFAULT_CHARSET = "utf-8"
APPEND_SLASH = False
DATA_UPLOAD_MAX_MEMORY_SIZE = 1_048_576
