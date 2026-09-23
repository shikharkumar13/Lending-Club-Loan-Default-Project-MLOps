# Serving image: model + API only.
#
# It installs the core dependencies (104 packages), not the training extra
# (940). MLflow, DVC, SHAP and the plotting libraries cannot run in production
# anyway, and every package shipped is extra weight and extra attack surface.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
WORKDIR /app

# Dependencies first, in their own layer: they change far less often than the
# code, so rebuilds after a code edit reuse this cached layer.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime

# libgomp is LightGBM's OpenMP runtime; the slim image does not include it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user: a container that only scores loans has no reason to
# have root inside its own filesystem.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY params.yaml ./
COPY artifacts/bundle ./artifacts/bundle

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "lending_club.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
