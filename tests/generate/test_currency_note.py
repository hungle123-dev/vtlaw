"""Evidence spanning two decrees must not let the model cite the superseded one.

Measured on 31 QA_NLP questions whose retrieved evidence contains two penalty
decrees: 15 answers cited only 100/2019/NĐ-CP while 168/2024/NĐ-CP sat in the
same evidence (0.484). Naming the newest decree explicitly in the prompt took
that to 7 (0.226). See scripts/measure_citation_currency.py.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

from vtlaw.config import Settings
from vtlaw.generate.answer import AnswerGenerator
from vtlaw.retrieve.search import Hit


def _hit(doc: str, uid: str) -> Hit:
    return Hit(uid=uid, score=1.0, doc_identity=doc, label="Clause", content="Nội dung")


OLD = _hit("100/2019/NĐ-CP", "100/2019/NĐ-CP::article::5::clause::2")
NEW = _hit("168/2024/NĐ-CP", "168/2024/NĐ-CP::article::6::clause::3")


def _client_returning(rows: list[dict]) -> MagicMock:
    client = MagicMock()
    session = client.session.return_value.__enter__.return_value
    session.run.return_value.data.return_value = rows
    return client


def _generator(client: MagicMock, llm: MagicMock) -> AnswerGenerator:
    return AnswerGenerator(client, MagicMock(), llm, Settings(neo4j_password="test"))


def test_prompt_names_the_newest_decree_when_evidence_spans_two():
    client = _client_returning([
        {"ident": "100/2019/NĐ-CP", "effect_date": "2020-01-01"},
        {"ident": "168/2024/NĐ-CP", "effect_date": "2025-01-01"},
    ])
    llm = MagicMock()
    llm.complete.return_value = "Theo Khoản 3 Điều 6 Nghị định 168/2024/NĐ-CP, bị phạt."

    with patch("vtlaw.generate.answer.build_full_context", return_value={}):
        _generator(client, llm).generate_from_hits(
            "Vượt đèn đỏ phạt bao nhiêu?", [OLD, NEW], as_of=date(2026, 8, 20)
        )

    prompt = llm.complete.call_args.kwargs["messages"][1]["content"]
    assert "168/2024/NĐ-CP có hiệu lực từ 2025-01-01 là văn bản mới nhất" in prompt
    # The older decree stays visible — it may be the only one covering a behaviour.
    assert "100/2019/NĐ-CP (hiệu lực 2020-01-01)" in prompt


def test_no_currency_note_when_the_evidence_has_one_decree():
    """With a single source there is nothing older to mistakenly prefer."""
    client = _client_returning([{"ident": "168/2024/NĐ-CP", "effect_date": "2025-01-01"}])
    llm = MagicMock()
    llm.complete.return_value = "Theo Khoản 3 Điều 6 Nghị định 168/2024/NĐ-CP, bị phạt."

    with patch("vtlaw.generate.answer.build_full_context", return_value={}):
        _generator(client, llm).generate_from_hits("Câu hỏi", [NEW])

    prompt = llm.complete.call_args.kwargs["messages"][1]["content"]
    assert "văn bản mới nhất" not in prompt
    # No decree comparison means no reason to query the graph for effect dates.
    client.session.assert_not_called()


def test_a_law_and_a_decree_are_not_compared_as_two_versions_of_one_rule():
    """Luật 36/2024/QH15 does not supersede a Nghị định; they are different instruments."""
    client = _client_returning([])
    llm = MagicMock()
    llm.complete.return_value = "Theo Khoản 3 Điều 6 Nghị định 168/2024/NĐ-CP, bị phạt."
    law = _hit("36/2024/QH15", "36/2024/QH15::article::11::clause::1")

    with patch("vtlaw.generate.answer.build_full_context", return_value={}):
        _generator(client, llm).generate_from_hits("Câu hỏi", [law, NEW])

    prompt = llm.complete.call_args.kwargs["messages"][1]["content"]
    assert "văn bản mới nhất" not in prompt
