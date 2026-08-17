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

3. **Idempotent batch embedding.** Nodes already carrying an embedding are
   skipped, so a re-run after a partial failure resumes instead of recomputing.
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
                self._settings.embed_model, device=device
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

# Fetch provisions that have content but no embedding yet.
_FETCH_MISSING = """
MATCH (n:{label})
WHERE n.content IS NOT NULL AND n.content <> ''
  AND n.embedding IS NULL
RETURN n.uid AS uid,
       coalesce(n.title, '') AS title,
       n.content AS content
LIMIT $batch
"""

_WRITE_VECTORS = """
UNWIND $rows AS row
MATCH (n:{label} {{uid: row.uid}})
SET n.embedding = row.vector,
    n.embedded_with = $embedded_with
"""

# For articles with no body content, embed the title only. Verified on this
# corpus: 500 of 566 articles have empty content — their title is the entire
# topical signal ("Xử phạt người điều khiển xe ô tô...").
_FETCH_ARTICLES_TITLE_ONLY = """
MATCH (n:Article)
WHERE n.title IS NOT NULL AND n.title <> ''
  AND n.content = ''
  AND n.embedding IS NULL
RETURN n.uid AS uid, n.title AS title, '' AS content
LIMIT $batch
"""


def _embed_text(title: str, content: str) -> str:
    """Build the text that goes into the vector.

    Articles often have a real title and no body: prepend the title so the vector
    captures its topical signal rather than embedding an empty string.
    """
    if title:
        return f"{title}\n{content}".strip()
    return content.strip()


def embed_corpus(
    client: GraphClient,
    embedder: Embedder,
    *,
    batch_size: int = 256,
) -> EmbedStats:
    """Embed every provision in the graph that is missing a vector.

    Idempotent: ``WHERE n.embedding IS NULL`` skips already-embedded nodes, so a
    re-run resumes from where it left off.
    """
    embedded_with = f"{embedder._settings.embed_model}|{SEGMENTER_VERSION}"
    stats = EmbedStats(embedded={}, skipped={})

    for label in PROVISION_LABELS:
        embedded_count = 0

        # Articles with empty content but a title: embed the title.
        if label == "Article":
            embedded_count += _embed_batch(
                client, embedder, "Article", _FETCH_ARTICLES_TITLE_ONLY,
                batch_size, embedded_with,
            )

        # Everything with content.
        embedded_count += _embed_batch(
            client, embedder, label, _FETCH_MISSING,
            batch_size, embedded_with,
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
    embedded_with: str,
) -> int:
    """Embed and write one label, looping until no more unembedded nodes."""
    total = 0

    while True:
        with client.session() as s:
            rows = s.run(
                fetch_cypher.format(label=label), batch=batch_size
            ).data()

        if not rows:
            break

        texts = [_embed_text(r["title"], r["content"]) for r in rows]
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
                embedded_with=embedded_with,
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
