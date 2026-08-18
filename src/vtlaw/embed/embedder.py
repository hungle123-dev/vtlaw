"""Generate vector embeddings for provisions and store them in Neo4j.

Three things this module gets right that a naive wrapper does not:

1. **Vietnamese word segmentation.** The model (PhoBERT-based) expects
   underscore-segmented text: ``người điều khiển`` becomes ``người_điều_khiển``.
   Without it, recall degrades silently. Segmentation is applied identically to
   indexed text and to queries.

2. **The real token limit.** The tokenizer config reports ``model_max_length``
   of ~1e30, a sentinel. The real limit is 256 tokens from
   ``sentence_bert_config.json``. We set it explicitly so long provisions are
   not silently truncated inside the encoder.

3. **Provenance-aware batch embedding.** Nodes are recomputed when the model
   or embedding-text transform changes, so stored vectors cannot silently use
   an older pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pyvi import ViTokenizer
from sentence_transformers import SentenceTransformer

from vtlaw.config import EMBED_DIM, EMBED_MAX_TOKENS, Settings
from vtlaw.graph.client import GraphClient
from vtlaw.graph.schema import PROVISION_LABELS

log = logging.getLogger(__name__)

SEGMENTER_VERSION = "pyvi-ViTokenizer"
EMBEDDING_TEXT_VERSION = "v2"


def embedding_fingerprint(settings: Settings) -> str:
    """Identity of the model and text transform used for stored vectors."""
    return (
        f"{settings.embed_model}@{settings.embed_model_revision}|"
        f"{SEGMENTER_VERSION}|{EMBEDDING_TEXT_VERSION}"
    )


def segment(text: str) -> str:
    """Word-segment Vietnamese for the PhoBERT tokenizer.

    Idempotent: re-running on already-segmented text (underscores present)
    leaves it intact, so calling this on a query that was pre-segmented is safe.
    """
    return ViTokenizer.tokenize(text)


@dataclass
class EmbedStats:
    """Result of an embedding run, per label."""

    embedded: dict[str, int]      # label -> count embedded this run
    skipped: dict[str, int]       # label -> count already embedded
    failed: list[str] = None      # uids that errored

    def __post_init__(self) -> None:
        if self.failed is None:
            self.failed = []

    def summary(self) -> str:
        parts = [
            f"{label}={self.embedded.get(label, 0)}"
            for label in PROVISION_LABELS
        ]
        skipped = [f"{label}={self.skipped.get(label, 0)}" for label in PROVISION_LABELS]
        return (
            f"embedded {', '.join(parts)}; "
            f"already_had {', '.join(skipped)}"
        )


class Embedder:
    """Wraps the sentence-transformers model with Vietnamese segmentation."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy model load. Importing this module must not load torch."""
        if self._model is None:
            device = self._settings.resolved_device()
            log.info(
                "loading %s on %s (dim=%d, max_tokens=%d)",
                self._settings.embed_model, device, EMBED_DIM, EMBED_MAX_TOKENS,
            )
            self._model = SentenceTransformer(
                self._settings.embed_model,
                revision=self._settings.embed_model_revision,
                device=device,
            )
            self._model.max_seq_length = EMBED_MAX_TOKENS
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Segment and embed a batch of texts.

        Returns unit-normalised vectors: the cosine index expects them, and it
        keeps the similarity score in a meaningful range.
        """
        if not texts:
            return []
        prepared = [segment(t) for t in texts]
        # Truncate long segments: model has 256-token limit (from
        # sentence_bert_config.json, not the ~1e30 sentinel in tokenizer_config).
        # We approximate by word count: each underscore-separated word is roughly
        # one token after Vietnamese word-segmentation.
        max_words = EMBED_MAX_TOKENS
        truncated = [
            " ".join(p.split()[:max_words]) if len(p.split()) > max_words else p
            for p in prepared
        ]
        vectors = self.model.encode(
            truncated,
            batch_size=self._settings.embed_batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(truncated) > 256,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def encode_query(self, text: str) -> list[float]:
        """Embed a single query string for vector search."""
        return self.encode([text])[0]


# ---------------------------------------------------------------------------
# Neo4j write-back
# ---------------------------------------------------------------------------

# Fetch provisions that need a vector from the current embedding pipeline.
# These are intentionally separate queries: optional matches for a Point's
# parents can cross-join when `n` is an Article or Clause.
_FETCH_ARTICLES = """
MATCH (n:Article)
WHERE n.content IS NOT NULL AND n.content <> ''
  AND (n.embedding IS NULL OR coalesce(n.embedding_fingerprint, '') <> $fingerprint)
RETURN n.uid AS uid,
       coalesce(n.title, '') AS title,
       n.content AS content,
       '' AS parent_content
LIMIT $batch
"""

_FETCH_CLAUSES = """
MATCH (article:Article)-[:HAS_CLAUSE]->(n:Clause)
WHERE n.content IS NOT NULL AND n.content <> ''
  AND (n.embedding IS NULL OR coalesce(n.embedding_fingerprint, '') <> $fingerprint)
RETURN n.uid AS uid,
       coalesce(article.title, '') AS title,
       n.content AS content,
       '' AS parent_content
LIMIT $batch
"""

_FETCH_POINTS = """
MATCH (article:Article)-[:HAS_CLAUSE]->(clause:Clause)-[:HAS_POINT]->(n:Point)
WHERE n.content IS NOT NULL AND n.content <> ''
  AND (n.embedding IS NULL OR coalesce(n.embedding_fingerprint, '') <> $fingerprint)
RETURN n.uid AS uid,
       coalesce(article.title, '') AS title,
       n.content AS content,
       clause.content AS parent_content
LIMIT $batch
"""

_FETCH_CONTENT = {
    "Article": _FETCH_ARTICLES,
    "Clause": _FETCH_CLAUSES,
    "Point": _FETCH_POINTS,
}

_WRITE_VECTORS = """
UNWIND $rows AS row
MATCH (n:{label} {{uid: row.uid}})
SET n.embedding = row.vector,
    n.embedded_with = $fingerprint,
    n.embedding_fingerprint = $fingerprint
"""

# For articles with no body content, embed the title only. Verified on this
# corpus: 500 of 566 articles have empty content — their title is the entire
# topical signal ("Xử phạt người điều khiển xe ô tô...").
_FETCH_ARTICLES_TITLE_ONLY = """
MATCH (n:Article)
WHERE n.title IS NOT NULL AND n.title <> ''
  AND n.content = ''
  AND (n.embedding IS NULL OR coalesce(n.embedding_fingerprint, '') <> $fingerprint)
RETURN n.uid AS uid, n.title AS title, '' AS content, '' AS parent_content
LIMIT $batch
"""


def _embed_text(title: str, content: str, parent_content: str = "") -> str:
    """Build the text that goes into the vector.

    Clauses inherit their article title. Points also include their parent clause,
    but keep their own offence first so the most specific signal survives token
    truncation. Articles with no body use their title as the topical signal.
    """
    if parent_content:
        return "\n".join(part for part in (content, parent_content, title) if part)
    return "\n".join(part for part in (title, content) if part)


def embed_corpus(
    client: GraphClient,
    embedder: Embedder,
    *,
    batch_size: int = 256,
) -> EmbedStats:
    """Embed every provision not produced by the current vector pipeline.

    Idempotent for a fingerprint: a re-run resumes after a partial failure, while
    a model or text-pipeline change deliberately refreshes every stale vector.
    """
    fingerprint = embedding_fingerprint(embedder._settings)
    stats = EmbedStats(embedded={}, skipped={})

    for label in PROVISION_LABELS:
        embedded_count = 0

        # Articles with empty content but a title: embed the title.
        if label == "Article":
            embedded_count += _embed_batch(
                client, embedder, "Article", _FETCH_ARTICLES_TITLE_ONLY,
                batch_size, fingerprint,
            )

        # Everything with content.
        embedded_count += _embed_batch(
            client, embedder, label, _FETCH_CONTENT[label],
            batch_size, fingerprint,
        )

        stats.embedded[label] = embedded_count

        # Count how many already had a vector (for honest reporting).
        # We just embedded `embedded_count` nodes, so the rest with embeddings
        # were already there before this run.
        with client.session() as s:
            now_embedded = s.run(
                f"MATCH (n:{label}) WHERE n.embedding IS NOT NULL "
                f"RETURN count(n) AS n"
            ).single()["n"]
        stats.skipped[label] = now_embedded - embedded_count

    return stats


def _embed_batch(
    client: GraphClient,
    embedder: Embedder,
    label: str,
    fetch_cypher: str,
    batch_size: int,
    fingerprint: str,
) -> int:
    """Embed and write one label, looping until no more unembedded nodes."""
    total = 0

    while True:
        with client.session() as s:
            rows = s.run(
                fetch_cypher.format(label=label), batch=batch_size, fingerprint=fingerprint
            ).data()

        if not rows:
            break

        texts = [
            _embed_text(r["title"], r["content"], r["parent_content"])
            for r in rows
        ]
        vectors = embedder.encode(texts)

        assert all(len(v) == EMBED_DIM for v in vectors), (
            f"vector width mismatch: expected {EMBED_DIM}"
        )

        payload = [
            {"uid": r["uid"], "vector": v}
            for r, v in zip(rows, vectors, strict=True)
        ]
        with client.session() as s:
            s.run(
                _WRITE_VECTORS.format(label=label),
                rows=payload,
                fingerprint=fingerprint,
            )

        total += len(rows)
        log.info("%s: embedded %d (total %d)", label, len(rows), total)

    return total


def embedding_coverage(client: GraphClient) -> list[dict]:
    """Per-label: total provisions, how many have content, how many have vectors.

    Used by the CLI to report honestly instead of assuming everything worked.
    """
    out: list[dict] = []
    with client.session() as s:
        for label in PROVISION_LABELS:
            rec = s.run(
                f"""
                MATCH (n:{label})
                WITH count(n) AS total,
                     sum(CASE WHEN n.content IS NOT NULL AND n.content <> ''
                              THEN 1 ELSE 0 END) AS with_content,
                     sum(CASE WHEN n.title IS NOT NULL AND n.title <> ''
                              AND (n.content IS NULL OR n.content = '')
                              THEN 1 ELSE 0 END) AS title_only,
                     sum(CASE WHEN n.embedding IS NOT NULL THEN 1 ELSE 0 END) AS embedded
                RETURN total, with_content, title_only, embedded
                """
            ).single()
            out.append({"label": label, **dict(rec)})
    return out
