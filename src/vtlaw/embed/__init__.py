"""Stage 4 — embed provisions into vectors and store them in Neo4j.

Reads the graph produced by :mod:`vtlaw.graph`, writes embeddings back onto the
same nodes. No network beyond the database and the model download.
"""

from vtlaw.embed.embedder import (
    EMBEDDING_TEXT_VERSION,
    SEGMENTER_VERSION,
    Embedder,
    EmbedStats,
    embed_corpus,
    embedding_coverage,
    embedding_fingerprint,
    segment,
)

__all__ = [
    "EmbedStats",
    "EMBEDDING_TEXT_VERSION",
    "Embedder",
    "SEGMENTER_VERSION",
    "embed_corpus",
    "embedding_coverage",
    "embedding_fingerprint",
    "segment",
]
