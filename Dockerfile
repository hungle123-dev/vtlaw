# Multi-stage: the ML wheels (torch, sentence-transformers) pull in build tooling
# and CUDA stubs that the running service never needs. Building in one stage and
# copying only the installed venv keeps that out of the shipped layer.
FROM python:3.12-slim AS builder

# uv resolves and installs far faster than pip here, and the project already
# pins through pyproject.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /bin/uv

WORKDIR /build
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Dependency layer first: source changes are frequent, dependencies are not, so
# this layer stays cached across ordinary code edits.
COPY pyproject.toml README.md ./
COPY src/ ./src/

# CPU-only torch. The default wheel carries ~2GB of CUDA libraries that are dead
# weight without a GPU, and the image is meant to run anywhere.
RUN uv venv /opt/venv \
 && VIRTUAL_ENV=/opt/venv uv pip install \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      --index-strategy unsafe-best-match \
      -e ".[embed,serve]"


FROM python:3.12-slim AS runtime

# curl is for the healthcheck below; nothing else is added.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Run as a non-root user. A container process that does not need to write outside
# its own cache should not be able to.
RUN useradd --create-home --uid 10001 vtlaw

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=vtlaw:vtlaw src/ ./src/
COPY --chown=vtlaw:vtlaw data/ ./data/
COPY --chown=vtlaw:vtlaw pyproject.toml README.md ./

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/vtlaw/.cache/huggingface \
    API_HOST=0.0.0.0 \
    API_PORT=18080

USER vtlaw

# The embedding model is ~500MB and downloaded on first use. Baking it into the
# image would make the build reproducible but triples its size; the tradeoff here
# is a slower first request in exchange for a lean image. Mount HF_HOME as a
# volume to keep the download across container restarts.

EXPOSE 18080

# /health reports 503 when Neo4j is unreachable, so this distinguishes "process
# alive" from "actually able to answer" — the whole point of the endpoint.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:18080/health || exit 1

CMD ["python", "-m", "vtlaw.cli", "api", "serve"]
