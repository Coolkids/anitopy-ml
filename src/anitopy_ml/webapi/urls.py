"""本地 API 路由。"""

from django.urls import path

from anitopy_ml.webapi.views import health_view, parse_batch_view, parse_view

urlpatterns = [
    path("healthz", health_view, name="health"),
    path("v1/parse", parse_view, name="parse"),
    path("v1/parse-batch", parse_batch_view, name="parse-batch"),
]
