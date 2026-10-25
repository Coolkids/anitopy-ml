"""供 Gunicorn 加载的 Django WSGI 入口。"""

from __future__ import annotations

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "anitopy_ml.webapi.settings")

from django.core.wsgi import get_wsgi_application

from anitopy_ml.webapi.service import startup_processing_config

application = get_wsgi_application()
startup_processing_config()
