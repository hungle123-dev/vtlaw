"""Stage 4 — embed provisions into vectors and store them in Neo4j.

Reads the graph produced by :mod:`vtlaw.graph`, writes embeddings back onto the
same nodes. No network beyond the database and the model download.
"""

from vtlaw.embed.embedder import (
    SEGMENTER_VERSION,
    Embedder,
    EmbedStats,
    embed_corpus,
    embedding_coverage,
    segment,
)

__all__ = [
    "EmbedStats",
    "Embedder",
    "SEGMENTER_VERSION",
    "embed_corpus",
    "embedding_coverage",
    "segment",
]
