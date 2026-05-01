# syntax=docker/dockerfile:1.7

# ----- Stage 1: builder ------------------------------------------------------
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --upgrade pip setuptools wheel

# Install torch from the CUDA 12.1 wheel index, then the rest from PyPI.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.0 \
    && pip install -r /tmp/requirements.txt \
    && python -m spacy download en_core_web_sm

# ----- Stage 2: runtime ------------------------------------------------------
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/app/models_cache \
    TRANSFORMERS_CACHE=/app/models_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy application code (adapters and cache come in at runtime via volumes).
COPY app /app/app
COPY scripts /app/scripts

RUN mkdir -p /app/models_cache /app/adapters \
    && useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
