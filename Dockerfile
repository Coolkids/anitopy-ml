# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.15 AS uv

FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

ARG DOWNLOAD_MODEL=1
ARG MODEL_URL="https://github.com/Coolkids/anitopy-ml/releases/download/v0.1.0/anitopy-ml-v11.zip"
ARG BASE_MODEL="FacebookAI/xlm-roberta-base"
ARG MODEL_DIR="/opt/anitopy-ml/models/anitopy-ml-v11"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/opt/anitopy-ml/huggingface \
    ANITOPY_MODEL_DIR=${MODEL_DIR} \
    ANITOPY_DEVICE=auto \
    ANITOPY_MAX_BATCH_SIZE=100 \
    ANITOPY_MAX_TITLE_LENGTH=4096

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev --extra inference --extra serve

RUN if [ "$DOWNLOAD_MODEL" = "1" ]; then \
      apt-get update && apt-get install --no-install-recommends -y curl unzip && \
      mkdir -p "$MODEL_DIR" "$HF_HOME" && \
      curl --fail --location --retry 5 --output /tmp/model.zip "$MODEL_URL" && \
      unzip -q /tmp/model.zip -d "$MODEL_DIR" && \
      test -f "$MODEL_DIR/best_model.pt" && \
      /app/.venv/bin/python -c "from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained('$BASE_MODEL'); AutoModel.from_pretrained('$BASE_MODEL')" && \
      rm -f /tmp/model.zip && \
      apt-get purge -y --auto-remove curl unzip && rm -rf /var/lib/apt/lists/*; \
    elif [ "$DOWNLOAD_MODEL" != "0" ]; then \
      echo "DOWNLOAD_MODEL 只能为0或1。" >&2; exit 1; \
    fi

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=45s --retries=3 \
  CMD /app/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=10)" || exit 1

CMD ["/app/.venv/bin/gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "120", "anitopy_ml.webapi.wsgi:application"]
