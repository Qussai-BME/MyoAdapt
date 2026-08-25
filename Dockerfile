# syntax=docker/dockerfile:1
# MyoAdapt research-only runtime image.
# Deploy behind a TLS reverse proxy with request-size and rate limits.

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY myoadapt/ ./myoadapt/
RUN python -m pip install --upgrade pip \
    && python -m pip install ".[ml,onnx,explain,tracking]"

FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="MyoAdapt" \
      org.opencontainers.image.description="Research-only CPU-native sEMG pattern-recognition software" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.source="https://github.com/Qussai-BME/MyoAdapt"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/app \
    MYOADAPT_REQUIRE_API_KEY=true

# Run the service without root privileges. Mount writable data/model/result
# directories explicitly and grant them to this UID at deployment time.
RUN groupadd --gid 10001 myoadapt \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /app --shell /usr/sbin/nologin myoadapt

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --chown=myoadapt:myoadapt pyproject.toml ./
COPY --chown=myoadapt:myoadapt myoadapt/ ./myoadapt/
COPY --chown=myoadapt:myoadapt data/sample/ ./data/sample/
RUN mkdir -p /app/data /app/models /app/results /app/experiments /app/reports \
    && chown -R myoadapt:myoadapt /app

USER myoadapt
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).status == 200 else 1)"

# Model paths and keys are operator-supplied server configuration only.
CMD ["uvicorn", "myoadapt.api.rest:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
