"""Embedding provenance must change when the model or text pipeline changes."""

from datetime import date

import pytest

from vtlaw.config import Settings
from vtlaw.embed import embed_corpus, embedding_fingerprint
from vtlaw.embed.embedder import _FETCH_CONTENT, _embed_text
from vtlaw.graph import GraphClient, import_documents
from vtlaw.parse import Document, parse_text


def _settings(**overrides: str) -> Settings:
    return Settings(neo4j_password="test-password", **overrides)


def test_embedding_fingerprint_changes_with_model_revision():
    first = embedding_fingerprint(_settings(embed_model_revision="revision-one"))
    second = embedding_fingerprint(_settings(embed_model_revision="revision-two"))

    assert first != second
    assert "revision-one" in first


def test_point_embedding_keeps_the_offence_before_its_legal_context():
    text = _embed_text(
        "Xử phạt xe ô tô",
        "Vượt quá tốc độ từ 05 km/h đến dưới 10 km/h",
        "Phạt tiền từ 800.000 đồng đến 1.000.000 đồng",
    )

    assert text.splitlines() == [
        "Vượt quá tốc độ từ 05 km/h đến dưới 10 km/h",
        "Phạt tiền từ 800.000 đồng đến 1.000.000 đồng",
        "Xử phạt xe ô tô",
    ]


def test_contextual_embedding_queries_cannot_cross_join_provision_levels():
    """A Point-parent match must not multiply Article rows during reindexing."""
    assert all("OPTIONAL MATCH" not in query for query in _FETCH_CONTENT.values())
    assert "(article:Article)-[:HAS_CLAUSE]->(n:Clause)" in _FETCH_CONTENT["Clause"]
    assert (
        "(article:Article)-[:HAS_CLAUSE]->(clause:Clause)-[:HAS_POINT]->(n:Point)"
        in _FETCH_CONTENT["Point"]
    )


@pytest.fixture
def clean_graph():
    graph = GraphClient()
    try:
        graph.verify()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j not reachable: {exc}")
    graph.wipe()
    yield graph
    graph.wipe()
    graph.close()


@pytest.mark.integration
def test_embed_corpus_replaces_vectors_with_an_old_fingerprint(clean_graph):
    settings = _settings(embed_model_revision="revision-one")
    document = Document(
        doc_guid="fingerprint-test",
        doc_identity="fingerprint-test",
        doc_name="Fingerprint test",
        effect_date=date(2025, 1, 1),
    )
    import_documents(
        clean_graph,
        [parse_text("Điều 1. Tiêu đề\n1. Nội dung\na) Chi tiết\n", document)],
    )
    with clean_graph.session() as session:
        session.run(
            "MATCH (n) WHERE n.uid IS NOT NULL "
            "SET n.embedding = $vector, n.embedding_fingerprint = 'old'",
            vector=[1.0] * 768,
        )

    class FakeEmbedder:
        _settings = settings

        @staticmethod
        def encode(texts: list[str]) -> list[list[float]]:
            return [[0.0] * 768 for _ in texts]

    embed_corpus(clean_graph, FakeEmbedder(), batch_size=10)

    with clean_graph.session() as session:
        fingerprints = session.run(
            "MATCH (n) WHERE n.uid IS NOT NULL "
            "RETURN DISTINCT n.embedding_fingerprint AS fingerprint"
        ).value("fingerprint")

    assert fingerprints == [embedding_fingerprint(settings)]
