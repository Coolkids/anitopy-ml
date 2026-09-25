# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.15 AS uv

FROM python:3.14-slim

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/opt/anitopy-ml/huggingface \
    ANITOPY_MODEL_DIR=/opt/anitopy-ml/models/anitopy-ml-v11 \
    ANITOPY_MODEL_URL=https://github.com/Coolkids/anitopy-ml/releases/download/v0.1.0/anitopy-ml-v11.zip \
    ANITOPY_MODEL_UPDATE=missing \
    ANITOPY_BASE_MODEL=FacebookAI/xlm-roberta-base \
    ANITOPY_DEVICE=auto \
    ANITOPY_MAX_BATCH_SIZE=100 \
    ANITOPY_MAX_TITLE_LENGTH=4096

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev --extra inference --extra serve

COPY docker/entrypoint.py /usr/local/bin/anitopy-ml-entrypoint

RUN chmod 755 /usr/local/bin/anitopy-ml-entrypoint && \
    mkdir -p /opt/anitopy-ml/models /opt/anitopy-ml/huggingface

VOLUME ["/opt/anitopy-ml/models", "/opt/anitopy-ml/huggingface"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
  CMD /app/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=10)" || exit 1

ENTRYPOINT ["/app/.venv/bin/python", "/usr/local/bin/anitopy-ml-entrypoint"]
CMD ["/app/.venv/bin/gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "120", "anitopy_ml.webapi.wsgi:application"]
