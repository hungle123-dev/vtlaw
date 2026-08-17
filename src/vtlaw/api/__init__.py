"""Stage 7 — HTTP API surface over retrieve + generate.

FastAPI app exposing:
    POST /chat    — answer a question with citations
    GET  /health  — check service readiness
"""

from vtlaw.api.app import app

__all__ = ["app"]
