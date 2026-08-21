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

    result = citation_search(
        client,
        "Khoản 3 Điều 6 168/2024/NĐ-CP",
        as_of=date(2025, 1, 1),
    )

    assert result.had_full_citation is True
    assert [hit.uid for hit in result.hits] == [
        "168/2024/NĐ-CP::article::6::clause::3"
    ]
    assert session.run.call_args.kwargs["uid"] == "168/2024/NĐ-CP::article::6::clause::3"
    assert session.run.call_args.kwargs["as_of"] == "2025-01-01"


def test_unknown_full_citation_is_recognized_separately_from_its_empty_hits():
    session = MagicMock()
    session.run.return_value.data.return_value = []
    client = MagicMock()
    client.session.return_value.__enter__.return_value = session

    result = citation_search(client, "Điểm a Khoản 3 Điều 999 999/2099/NĐ-CP")

    assert result.had_full_citation is True
    assert result.hits == []
