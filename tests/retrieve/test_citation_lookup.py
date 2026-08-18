"""Exact legal citations should bypass approximate retrieval."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from vtlaw.retrieve.search import citation_search, resolve_citation


def test_resolve_citation_uses_the_most_specific_provision():
    target = resolve_citation(
        "Theo Điểm A Khoản 3 Điều 6 Nghị định 168/2024/NĐ-CP thì sao?"
    )

    assert target is not None
    assert target.label == "Point"
    assert target.uid == "168/2024/NĐ-CP::article::6::clause::3::point::a"


def test_citation_search_applies_the_requested_effective_date():
    session = MagicMock()
    session.run.return_value.data.return_value = [
        {
            "uid": "168/2024/NĐ-CP::article::6::clause::3",
            "score": 1.0,
            "doc_identity": "168/2024/NĐ-CP",
            "content": "Phạt tiền.",
            "title": None,
        }
    ]
    client = MagicMock()
    client.session.return_value.__enter__.return_value = session

    hits = citation_search(
        client,
        "Khoản 3 Điều 6 168/2024/NĐ-CP",
        as_of=date(2025, 1, 1),
    )

    assert [hit.uid for hit in hits] == ["168/2024/NĐ-CP::article::6::clause::3"]
    assert session.run.call_args.kwargs["uid"] == "168/2024/NĐ-CP::article::6::clause::3"
    assert session.run.call_args.kwargs["as_of"] == "2025-01-01"
