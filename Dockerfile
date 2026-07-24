FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VSRX_ALLOWED_ROOTS=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/vsrx
COPY pyproject.toml README.md LICENSE NOTICE.md ./
COPY src ./src
COPY configs ./configs
COPY docs ./docs
RUN python -m pip install --upgrade pip \
    && python -m pip install '.[ocr]'

RUN useradd --create-home --uid 10001 vsrx \
    && mkdir -p /data/input /data/output /data/work /data/models \
    && chown -R vsrx:vsrx /data
USER vsrx
WORKDIR /data

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["vsrx", "serve", "--host", "0.0.0.0", "--port", "8765", "--profile", "balanced"]
