"""Tests for graph-enhanced context building.

Tests cover:
- fetch_hierarchy: walks UP from hit to Document, builds citation context
- fetch_sibling_points: walks SIDEWAYS to get sibling Points under same Clause
- fetch_children_context: walks DOWN to get descendant content
- build_full_context: combines all three strategies
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vtlaw.retrieve.context_builder import (
    build_full_context,
    fetch_children_context,
    fetch_hierarchy,
    fetch_sibling_points,
)
from vtlaw.retrieve.search import Hit


def make_hit(
    uid: str = "test::1",
    score: float = 0.5,
    doc_identity: str = "test",
    label: str = "Clause",
    content: str = "content",
) -> Hit:
    return Hit(
        uid=uid,
        score=score,
        doc_identity=doc_identity,
        label=label,  # type: ignore[arg-type]
        content=content,
    )


@pytest.fixture
def mock_client():
    return MagicMock()


def _node(label: str, **props) -> dict:
    """One hierarchy entry in the shape the Cypher projection actually returns.

    The query asks for `{labels: labels(n), props: properties(n)}` because
    `nodes(path)` read through Result.data() arrives as bare property dicts with
    no labels at all. Fixtures that flattened labels alongside the properties
    tested a shape Neo4j never sends, which hid a real bug: every label test fell
    through, so the parent Clause carrying the penalty amount never reached the
    prompt.
    """
    return {"labels": [label], "props": props}


def _mock_data(mock_client, data):
    """Helper to set mock Neo4j query data."""
    mock_client.session.return_value.__enter__.return_value.run.return_value.data.return_value = (  # noqa: E501
        data
    )


def _mock_side_effect(mock_client, effect):
    """Helper to set mock Neo4j query side effect."""
    mock_client.session.return_value.__enter__.return_value.run.side_effect = effect


# ---------------------------------------------------------------------------
# fetch_hierarchy (walk UP)
# ---------------------------------------------------------------------------


class TestFetchHierarchy:
    def test_empty_uids_returns_empty(self, mock_client):
        result = fetch_hierarchy(mock_client, [])
        assert result == {}

    def test_builds_full_path_with_content(self, mock_client):
        """Walks from Point up to Document, includes hierarchy + content."""
        _mock_data(mock_client, [
            {
                "uid": "168/2024/NĐ-CP::article::6::clause::3::point::a",
                "hierarchy": [
                    _node("Article", number="6", title="Vi phạm tốc độ"),
                    _node("Clause", number="3", content="Phạt tiền từ 800.000đ"),
                    _node("Point", letter="a", content="Chạy quá tốc độ 5-10 km/h"),
                ],
                "doc_identity": "168/2024/NĐ-CP",
                "effect_date": "2025-01-01",
            }
        ])

        result = fetch_hierarchy(mock_client, ["168/2024/NĐ-CP::article::6::clause::3::point::a"])

        assert len(result) == 1
        ctx = result["168/2024/NĐ-CP::article::6::clause::3::point::a"]
        assert "168/2024/NĐ-CP" in ctx
        assert "2025-01-01" in ctx
        assert "Điều 6" in ctx
        assert "Vi phạm tốc độ" in ctx
        assert "Khoản 3" in ctx
        assert "Điểm a" in ctx
        # The whole reason for walking UP: the Point names the offence, the
        # parent Clause carries the money. An answer built without it states a
        # penalty amount the evidence never contained.
        assert "Phạt tiền từ 800.000đ" in ctx

    def test_handles_article_without_title(self, mock_client):
        _mock_data(mock_client, [
            {
                "uid": "test::article::1",
                "hierarchy": [
                    _node("Article", number="1", title=None),
                ],
                "doc_identity": "test",
                "effect_date": None,
            }
        ])

        result = fetch_hierarchy(mock_client, ["test::article::1"])
        assert "Điều 1" in result["test::article::1"]

    def test_includes_main_content(self, mock_client):
        """The target's own content is appended after the hierarchy path."""
        _mock_data(mock_client, [
            {
                "uid": "test::clause::1",
                "hierarchy": [
                    _node("Clause", number="1", content="Main content"),
                ],
                "doc_identity": "test",
                "effect_date": "2025-01-01",
            }
        ])

        result = fetch_hierarchy(mock_client, ["test::clause::1"])
        assert "Nội dung: Main content" in result["test::clause::1"]


# ---------------------------------------------------------------------------
# fetch_sibling_points (walk SIDEWAYS)
# ---------------------------------------------------------------------------


class TestFetchSiblingPoints:
    def test_empty_uids_returns_empty(self, mock_client):
        result = fetch_sibling_points(mock_client, [])
        assert result == {}

    def test_returns_siblings_for_point(self, mock_client):
        """Fetches other Points under the same Clause."""
        _mock_data(mock_client, [
            {
                "uid": "test::point::a",
                "siblings": [
                    {"letter": "b", "content": "Second point"},
                    {"letter": "c", "content": "Third point"},
                ],
            }
        ])

        result = fetch_sibling_points(mock_client, ["test::point::a"])

        assert len(result) == 1
        ctx = result["test::point::a"]
        assert "Điểm b" in ctx
        assert "Điểm c" in ctx
        assert "Second point" in ctx
        assert "Third point" in ctx

    def test_ignores_points_without_siblings(self, mock_client):
        _mock_data(mock_client, [
            {
                "uid": "test::point::a",
                "siblings": [],
            }
        ])

        result = fetch_sibling_points(mock_client, ["test::point::a"])
        assert result == {}

    def test_sorts_siblings_by_letter(self, mock_client):
        _mock_data(mock_client, [
            {
                "uid": "test::point::b",
                "siblings": [
                    {"letter": "c", "content": "Third"},
                    {"letter": "a", "content": "First"},
                ],
            }
        ])

        result = fetch_sibling_points(mock_client, ["test::point::b"])
        ctx = result["test::point::b"]
        lines = ctx.split("\n")
        assert "Điểm a" in lines[0]
        assert "Điểm c" in lines[1]


# ---------------------------------------------------------------------------
# fetch_children_context (walk DOWN)
# ---------------------------------------------------------------------------


class TestFetchChildrenContext:
    def test_empty_uids_returns_empty(self, mock_client):
        result = fetch_children_context(mock_client, [])
        assert result == {}

    def test_returns_children_for_article(self, mock_client):
        """Fetches Clause children for an Article."""
        _mock_data(mock_client, [
            {
                "uid": "test::article::1",
                "target_label": "Article",
                "children": [
                    {"label": "Clause", "number": "1", "content": "First clause", "uid": "a1"},
                    {"label": "Clause", "number": "2", "content": "Second clause", "uid": "a2"},
                ],
            }
        ])

        result = fetch_children_context(mock_client, ["test::article::1"])

        assert len(result) == 1
        ctx = result["test::article::1"]
        assert "Khoản 1" in ctx
        assert "Khoản 2" in ctx
        assert "First clause" in ctx
        assert "Second clause" in ctx

    def test_returns_children_for_clause(self, mock_client):
        """Fetches Point children for a Clause."""
        _mock_data(mock_client, [
            {
                "uid": "test::clause::1",
                "target_label": "Clause",
                "children": [
                    {"label": "Point", "letter": "a", "content": "Point a", "uid": "p1"},
                    {"label": "Point", "letter": "b", "content": "Point b", "uid": "p2"},
                ],
            }
        ])

        result = fetch_children_context(mock_client, ["test::clause::1"])

        assert len(result) == 1
        ctx = result["test::clause::1"]
        assert "Điểm a" in ctx
        assert "Điểm b" in ctx

    def test_ignores_nodes_without_children(self, mock_client):
        _mock_data(mock_client, [
            {
                "uid": "test::article::1",
                "target_label": "Article",
                "children": [],
            }
        ])

        result = fetch_children_context(mock_client, ["test::article::1"])
        assert result == {}

    def test_ignores_point_nodes(self, mock_client):
        """Only Article/Clause have descendants; Point nodes are leaves."""
        _mock_data(mock_client, [
            {
                "uid": "test::point::a",
                "target_label": "Point",
                "children": [],
            }
        ])

        result = fetch_children_context(mock_client, ["test::point::a"])
        assert result == {}


# ---------------------------------------------------------------------------
# build_full_context (combine all three)
# ---------------------------------------------------------------------------


class TestBuildFullContext:
    def test_empty_hits_returns_empty(self, mock_client):
        result = build_full_context(mock_client, [])
        assert result == {}

    def test_combines_hierarchy_and_children(self, mock_client):
        """For an Article, combines hierarchy path with children context."""
        uid = "test::article::1"

        def mock_run(query, **kwargs):
            mock = MagicMock()
            if "nodes(path)" in query:
                mock.data.return_value = [
                    {
                        "uid": uid,
                        "hierarchy": [
                            _node("Article", number="1", title="Test"),
                        ],
                        "doc_identity": "test",
                        "effect_date": "2025-01-01",
                    }
                ]
            elif "HAS_CLAUSE" in query:
                mock.data.return_value = [
                    {
                        "uid": uid,
                        "target_label": "Article",
                        "children": [
                            {"label": "Clause", "number": "1", "content": "Clause content",
                             "uid": "c1"},
                        ],
                    }
                ]
            else:
                mock.data.return_value = []
            return mock

        mock_client.session.return_value.__enter__.return_value.run.side_effect = mock_run

        hits = [make_hit(uid=uid, label="Article")]
        result = build_full_context(mock_client, hits)

        assert len(result) == 1
        ctx = result[uid]
        assert "Điều 1" in ctx
        assert "Test" in ctx
        assert "Khoản 1" in ctx
        assert "Clause content" in ctx

    def test_retains_rendered_parent_clause_as_a_full_evidence_hit(self, mock_client):
        point_uid = "168/2024/NĐ-CP::article::6::clause::3::point::a"
        clause_uid = "168/2024/NĐ-CP::article::6::clause::3"

        def mock_run(query, **kwargs):
            mock = MagicMock()
            if "nodes(path)" in query:
                mock.data.return_value = [
                    {
                        "uid": point_uid,
                        "hierarchy": [
                            _node(
                                "Article",
                                uid="168/2024/NĐ-CP::article::6",
                                number="6",
                                title="Vi phạm tốc độ",
                            ),
                            _node(
                                "Clause",
                                uid=clause_uid,
                                number="3",
                                content="Phạt tiền từ 800.000 đồng",
                            ),
                            _node(
                                "Point",
                                uid=point_uid,
                                letter="a",
                                content="Chạy quá tốc độ",
                            ),
                        ],
                        "doc_identity": "168/2024/NĐ-CP",
                        "effect_date": "2025-01-01",
                    }
                ]
            else:
                mock.data.return_value = []
            return mock

        mock_client.session.return_value.__enter__.return_value.run.side_effect = mock_run
        point = make_hit(
            uid=point_uid,
            score=0.87,
            doc_identity="168/2024/NĐ-CP",
            label="Point",
            content="Chạy quá tốc độ",
        )
        rendered_evidence: dict[str, Hit] = {}

        build_full_context(
            mock_client,
            [point],
            rendered_evidence=rendered_evidence,
        )

        assert rendered_evidence[point_uid] == Hit(
            uid=point_uid,
            score=0.0,
            doc_identity="168/2024/NĐ-CP",
            label="Point",
            content="Chạy quá tốc độ",
        )
        assert rendered_evidence[clause_uid] == Hit(
            uid=clause_uid,
            score=0.0,
            doc_identity="168/2024/NĐ-CP",
            label="Clause",
            content="Phạt tiền từ 800.000 đồng",
        )

    def test_graph_context_replaces_the_raw_hit_record(self, mock_client):
        point_uid = "168/2024/NĐ-CP::article::6::clause::3::point::a"

        def mock_run(query, **kwargs):
            mock = MagicMock()
            if "nodes(path)" in query:
                mock.data.return_value = [
                    {
                        "uid": point_uid,
                        "hierarchy": [
                            _node(
                                "Point",
                                uid=point_uid,
                                letter="a",
                                content="Nội dung chuẩn từ đồ thị",
                            ),
                        ],
                        "doc_identity": "168/2024/NĐ-CP",
                        "effect_date": "2025-01-01",
                    }
                ]
            else:
                mock.data.return_value = []
            return mock

        mock_client.session.return_value.__enter__.return_value.run.side_effect = mock_run
        raw_hit = make_hit(
            uid=point_uid,
            score=0.87,
            doc_identity="168/2024/NĐ-CP",
            label="Point",
            content="Đoạn trích cũ từ retrieval",
        )
        rendered_evidence: dict[str, Hit] = {}

        result = build_full_context(
            mock_client,
            [raw_hit],
            rendered_evidence=rendered_evidence,
        )

        assert "Nội dung chuẩn từ đồ thị" in result[point_uid]
        assert rendered_evidence[point_uid] == Hit(
            uid=point_uid,
            score=0.0,
            doc_identity="168/2024/NĐ-CP",
            label="Point",
            content="Nội dung chuẩn từ đồ thị",
        )

    def test_records_only_rendered_article_children_as_evidence(self, mock_client):
        """A rendered child may be cited, but blank descendants may not."""
        article_uid = "168/2024/NĐ-CP::article::6"
        displayed_point = article_uid + "::clause::3::point::a"
        undisplayed_point = article_uid + "::clause::3::point::b"

        def mock_run(query, **kwargs):
            mock = MagicMock()
            if "nodes(path)" in query:
                mock.data.return_value = [
                    {
                        "uid": article_uid,
                        "hierarchy": [
                            _node("Article", uid=article_uid, number="6", title="Test"),
                        ],
                        "doc_identity": "168/2024/NĐ-CP",
                        "effect_date": "2025-01-01",
                    }
                ]
            elif "HAS_CLAUSE" in query:
                mock.data.return_value = [
                    {
                        "uid": article_uid,
                        "target_label": "Article",
                        "children": [
                            {
                                "label": "Point",
                                "letter": "a",
                                "content": "Nội dung hiển thị",
                                "uid": displayed_point,
                            },
                            {
                                "label": "Point",
                                "letter": "b",
                                "content": "",
                                "uid": undisplayed_point,
                            },
                        ],
                    }
                ]
            else:
                mock.data.return_value = []
            return mock

        mock_client.session.return_value.__enter__.return_value.run.side_effect = mock_run
        evidence_uids: set[str] = set()
        rendered_evidence: dict[str, Hit] = {}

        build_full_context(
            mock_client,
            [make_hit(uid=article_uid, label="Article")],
            evidence_uids=evidence_uids,
            rendered_evidence=rendered_evidence,
        )

        assert evidence_uids == {article_uid, displayed_point}
        assert rendered_evidence[displayed_point].label == "Point"
        assert undisplayed_point not in rendered_evidence

    def test_combines_hierarchy_and_siblings_for_point(self, mock_client):
        """For a Point, combines hierarchy with sibling points."""
        uid = "test::point::a"

        def mock_run(query, **kwargs):
            mock = MagicMock()
            if "nodes(path)" in query:
                mock.data.return_value = [
                    {
                        "uid": uid,
                        "hierarchy": [
                            _node("Point", letter="a", content="Point a"),
                        ],
                        "doc_identity": "test",
                        "effect_date": "2025-01-01",
                    }
                ]
            elif "sibling" in query:
                mock.data.return_value = [
                    {
                        "uid": uid,
                        "siblings": [
                            {"letter": "b", "content": "Point b"},
                        ],
                    }
                ]
            else:
                mock.data.return_value = []
            return mock

        mock_client.session.return_value.__enter__.return_value.run.side_effect = mock_run

        hits = [make_hit(uid=uid, label="Point")]
        result = build_full_context(mock_client, hits)

        assert len(result) == 1
        ctx = result[uid]
        assert "Điểm a" in ctx
        assert "Các điểm khác" in ctx
        assert "Điểm b" in ctx

    def test_records_rendered_sibling_point_as_evidence(self, mock_client):
        point_uid = "168/2024/NĐ-CP::article::6::clause::3::point::a"
        sibling_uid = "168/2024/NĐ-CP::article::6::clause::3::point::b"

        def mock_run(query, **kwargs):
            mock = MagicMock()
            if "nodes(path)" in query:
                mock.data.return_value = [
                    {
                        "uid": point_uid,
                        "hierarchy": [
                            _node("Point", uid=point_uid, letter="a", content="Điểm a"),
                        ],
                        "doc_identity": "168/2024/NĐ-CP",
                        "effect_date": "2025-01-01",
                    }
                ]
            elif "sibling" in query:
                mock.data.return_value = [
                    {
                        "uid": point_uid,
                        "siblings": [
                            {
                                "uid": sibling_uid,
                                "letter": "b",
                                "content": "Điểm b hiển thị",
                            }
                        ],
                    }
                ]
            else:
                mock.data.return_value = []
            return mock

        mock_client.session.return_value.__enter__.return_value.run.side_effect = mock_run
        evidence_uids: set[str] = set()
        rendered_evidence: dict[str, Hit] = {}

        build_full_context(
            mock_client,
            [make_hit(uid=point_uid, label="Point")],
            evidence_uids=evidence_uids,
            rendered_evidence=rendered_evidence,
        )

        assert evidence_uids == {point_uid, sibling_uid}
        assert rendered_evidence[sibling_uid].content == "Điểm b hiển thị"

    def test_falls_back_to_hit_content_when_no_context(self, mock_client):
        """If graph queries return nothing, uses the hit's own content."""
        _mock_data(mock_client, [])

        hits = [make_hit(uid="test::1", content="Fallback content")]
        rendered_evidence: dict[str, Hit] = {}
        result = build_full_context(
            mock_client,
            hits,
            rendered_evidence=rendered_evidence,
        )

        assert result["test::1"] == "Fallback content"
        assert rendered_evidence == {hits[0].uid: hits[0]}
