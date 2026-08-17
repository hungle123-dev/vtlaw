"""Unit tests for the parser — synthetic fixtures, no dataset needed.

Each test pins one rule that the corpus proved matters.
"""

from __future__ import annotations

from datetime import date

import pytest

from vtlaw.parse import (
    Document,
    parse_text,
    split_footer,
    uid_to_citation,
)
from vtlaw.parse.patterns import (
    RE_ARTICLE,
    RE_CLAUSE,
    RE_POINT,
    closes_quote,
    opens_quote,
    point_sort_key,
)


def doc(identity: str = "168/2024/NĐ-CP") -> Document:
    return Document(
        doc_guid="test-guid",
        doc_identity=identity,
        doc_name="Test document",
        effect_date=date(2025, 1, 1),
    )


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "number"),
    [
        ("Điều 6. Xử phạt", "6"),
        ("Điều 18a. Bổ sung", "18a"),
        ("Điều 58e. Vi phạm", "58e"),
        ("Điều 1 Quy định chung", "1"),  # no period after the number
    ],
)
def test_article_pattern_accepts_letter_suffix(line, number):
    match = RE_ARTICLE.match(line)
    assert match is not None
    assert match.group(1) == number


@pytest.mark.parametrize(
    ("line", "number"),
    [
        ("3. Phạt tiền từ 800.000 đồng", "3"),
        ("2a. Bổ sung nội dung", "2a"),
        ("18a. Sử dụng dữ liệu", "18a"),
        ("75. Khoản cuối", "75"),
    ],
)
def test_clause_pattern_accepts_letter_suffix(line, number):
    match = RE_CLAUSE.match(line)
    assert match is not None
    assert match.group(1) == number


def test_clause_pattern_is_bounded_to_two_digits():
    """A year or a long ordinal must not be read as a clause.

    The highest clause number in the corpus is 75, and no line begins with three
    or more digits followed by a period, so the bound is safe and it stops
    "2024. ..." from becoming clause 2024.
    """
    assert RE_CLAUSE.match("2024. Năm ban hành") is None
    assert RE_CLAUSE.match("100. Một trăm") is None
    assert RE_CLAUSE.match("75. Bảy mươi lăm") is not None


def test_clause_pattern_requires_whitespace_after_period():
    """`1.5 triệu đồng` is an amount, not clause 1."""
    assert RE_CLAUSE.match("1.5 triệu đồng") is None
    assert RE_CLAUSE.match("1. Nội dung") is not None


@pytest.mark.parametrize("letter", ["a", "b", "c", "d", "đ", "e", "g", "y"])
def test_point_pattern_covers_vietnamese_letters(letter):
    match = RE_POINT.match(f"{letter}) Nội dung")
    assert match is not None
    assert match.group(1) == letter


def test_point_ordering_places_d_stroke_after_d():
    """Vietnamese legal ordering is a, b, c, d, đ, e — not ASCII order, which
    would push "đ" past "y"."""
    letters = ["e", "đ", "a", "d", "b"]
    assert sorted(letters, key=point_sort_key) == ["a", "b", "d", "đ", "e"]


def test_point_sort_key_ignores_disambiguation_suffix():
    assert point_sort_key("a#2") == point_sort_key("a")


@pytest.mark.parametrize("line", ['"Điều 5. Nội dung', "“Điều 5. Nội dung"])
def test_opens_quote_detects_both_quote_styles(line):
    assert opens_quote(line)


@pytest.mark.parametrize("line", ['... nội dung."', "... nội dung.”", '... nội dung";'])
def test_closes_quote_strips_trailing_punctuation(line):
    """Quoted amendments usually end as `..."` or `...".;` so the closing mark is
    not the final character."""
    assert closes_quote(line)


# ---------------------------------------------------------------------------
# Quote-block tracking — the rule that decides what is NOT a provision
# ---------------------------------------------------------------------------


def test_hierarchy_inside_quotes_is_content_not_structure():
    """An amending document reproduces the text it inserts verbatim.

    Verified on the corpus: all 18 letter-suffixed headers (`Điều 58e`, `2a.`,
    `18a.`) appear inside quoted amendment text and zero appear as the document's
    own headings. Parsing them as provisions would fabricate articles that do not
    exist in the document being parsed.
    """
    text = """Điều 2. Sửa đổi, bổ sung
32. Bổ sung Điều 58e vào sau Điều 58d như sau:
“Mục 12. VI PHẠM QUY ĐỊNH VỀ LAO ĐỘNG HÀNG HẢI
Điều 58e. Vi phạm quy định về giao kết hợp đồng
1. Từ 2.000.000 đồng đến 5.000.000 đồng."
33. Khoản tiếp theo.
"""
    parsed = parse_text(text, doc("123/2021/NĐ-CP"))

    article_numbers = [p.number for p in parsed.by_level("article")]
    assert article_numbers == ["2"], "Điều 58e is quoted text, not a heading here"

    clause_numbers = [p.number for p in parsed.by_level("clause")]
    assert clause_numbers == ["32", "33"], "the quoted `1.` must not become a clause"

    quoted = parsed.by_level("clause")[0].content
    assert "Điều 58e" in quoted, "quoted text is kept as clause content"


def test_quote_closes_and_parsing_resumes():
    text = """Điều 1. Mở đầu
1. Sửa đổi như sau:
“Điều 99. Nội dung được chèn."
2. Khoản sau khi đóng ngoặc.
"""
    parsed = parse_text(text, doc())

    assert [p.number for p in parsed.by_level("article")] == ["1"]
    assert [p.number for p in parsed.by_level("clause")] == ["1", "2"]


def test_unclosed_quote_absorbs_remaining_lines():
    """A quote that never closes swallows the rest of the document.

    That is the safe direction: inventing provisions from unbalanced quotes would
    put text under citations it does not belong to.
    """
    text = """Điều 1. Mở đầu
1. Sửa đổi:
“Điều 99. Không đóng ngoặc
2. Dòng này nằm trong ngoặc chưa đóng
"""
    parsed = parse_text(text, doc())

    assert [p.number for p in parsed.by_level("clause")] == ["1"]


# ---------------------------------------------------------------------------
# Duplicate numbering
# ---------------------------------------------------------------------------


def test_duplicate_clause_numbering_keeps_both_provisions():
    """Real case: 118/2025/QH15 Điều 5 has two distinct `khoản 4`, each amending
    a different law. A UID of (doc, article, clause) merges them and loses legal
    text; a `#n` suffix keeps both.
    """
    text = """Điều 5. Sửa đổi
4. Sửa đổi khoản 4 Điều 25 như sau: nội dung A.
5. Sửa đổi Điều 27 như sau: nội dung B.
4. Sửa đổi điểm a khoản 1 Điều 29 như sau: nội dung C.
5. Sửa đổi điểm a khoản 4 Điều 30 như sau: nội dung D.
"""
    parsed = parse_text(text, doc("118/2025/QH15"))
    clauses = parsed.by_level("clause")

    assert len(clauses) == 4, "no provision may be dropped"
    assert len({c.uid for c in clauses}) == 4, "UIDs must stay distinct"

    uids = [c.uid for c in clauses]
    assert uids[0] == "118/2025/QH15::article::5::clause::4"
    assert uids[2] == "118/2025/QH15::article::5::clause::4#2"

    contents = [c.content for c in clauses]
    assert "nội dung A" in contents[0]
    assert "nội dung C" in contents[2], "the second `khoản 4` keeps its own text"


def test_duplicate_numbering_is_reported_not_hidden():
    text = "Điều 5. X\n4. A.\n4. B.\n"
    parsed = parse_text(text, doc("118/2025/QH15"))

    duplicates = [p for p in parsed.provisions if p.has_duplicate_numbering]
    assert len(duplicates) == 1
    assert duplicates[0].occurrence == 2


def test_citation_hides_the_disambiguation_suffix():
    """`#2` is an internal key. A legal citation never shows it."""
    uid = "118/2025/QH15::article::5::clause::4#2"
    assert uid_to_citation(uid) == "Khoản 4 Điều 5 118/2025/QH15"


# ---------------------------------------------------------------------------
# Hierarchy and parent linkage
# ---------------------------------------------------------------------------


def test_full_hierarchy_is_linked_by_parent_uid():
    text = """Chương I. QUY ĐỊNH CHUNG
Mục 1. HÌNH THỨC
Điều 6. Xử phạt người điều khiển xe ô tô
3. Phạt tiền từ 800.000 đồng đến 1.000.000 đồng:
a) Điều khiển xe chạy quá tốc độ từ 05 km/h đến dưới 10 km/h;
b) Không thắt dây an toàn;
"""
    parsed = parse_text(text, doc())

    article = parsed.by_level("article")[0]
    clause = parsed.by_level("clause")[0]
    points = parsed.by_level("point")

    assert article.uid == "168/2024/NĐ-CP::article::6"
    assert clause.parent_uid == article.uid
    assert all(p.parent_uid == clause.uid for p in points)
    assert [p.number for p in points] == ["a", "b"]


def test_point_carries_both_ancestor_numbers():
    text = "Điều 6. X\n3. Y:\na) Z;\n"
    parsed = parse_text(text, doc())
    point = parsed.by_level("point")[0]

    assert point.parent_article == "6"
    assert point.parent_clause == "3"
    assert point.citation == "Điểm a Khoản 3 Điều 6 168/2024/NĐ-CP"


def test_part_chapter_section_reset_article_context():
    """Organisational levels carry no retrievable text, but they must close the
    open article so a stray numbered line does not attach to the previous one."""
    text = """Điều 6. Trước
1. Nội dung.
Chương II. CHƯƠNG MỚI
Điều 7. Sau
1. Nội dung khác.
"""
    parsed = parse_text(text, doc())
    clauses = parsed.by_level("clause")

    assert [c.parent_article for c in clauses] == ["6", "7"]


def test_clause_before_any_article_is_not_a_provision():
    """A numbered line in the preamble is prose. Recording it as a clause would
    create a provision with no parent."""
    text = """Căn cứ Luật Xử lý vi phạm hành chính;
1. Đây là dòng liệt kê trong phần mở đầu.
Điều 1. Phạm vi
1. Nghị định này quy định.
"""
    parsed = parse_text(text, doc())
    clauses = parsed.by_level("clause")

    assert len(clauses) == 1
    assert clauses[0].parent_article == "1"
    assert "liệt kê trong phần mở đầu" in parsed.preamble


def test_point_before_any_clause_is_not_a_provision():
    text = "Điều 6. X\na) Không có khoản mẹ;\n1. Khoản thật:\nb) Điểm thật;\n"
    parsed = parse_text(text, doc())
    points = parsed.by_level("point")

    assert [p.number for p in points] == ["b"]
    assert points[0].parent_clause == "1"


# ---------------------------------------------------------------------------
# Content assembly
# ---------------------------------------------------------------------------


def test_continuation_lines_attach_to_the_deepest_open_provision():
    text = """Điều 6. Tiêu đề
1. Dòng đầu của khoản
dòng tiếp của khoản
a) Dòng đầu của điểm
dòng tiếp của điểm
"""
    parsed = parse_text(text, doc())

    clause = parsed.by_level("clause")[0]
    point = parsed.by_level("point")[0]

    assert clause.content == "Dòng đầu của khoản\ndòng tiếp của khoản"
    assert point.content == "Dòng đầu của điểm\ndòng tiếp của điểm"


def test_article_with_body_text_keeps_it_separate_from_title():
    text = "Điều 6. Tiêu đề điều\nNội dung riêng của điều.\n1. Khoản một.\n"
    parsed = parse_text(text, doc())
    article = parsed.by_level("article")[0]

    assert article.title == "Tiêu đề điều"
    assert article.content == "Nội dung riêng của điều."


def test_embedding_text_prepends_article_title():
    """Most articles have no body of their own; the title is the whole signal."""
    text = "Điều 6. Xử phạt người điều khiển xe ô tô\n1. Khoản.\n"
    parsed = parse_text(text, doc())
    article = parsed.by_level("article")[0]

    assert article.content == ""
    assert article.embedding_text() == "Xử phạt người điều khiển xe ô tô"


def test_ordinal_preserves_document_order():
    text = "Điều 1. A\n1. B\na) C\nĐiều 2. D\n"
    parsed = parse_text(text, doc())

    ordinals = [p.ordinal for p in parsed.provisions]
    assert ordinals == sorted(ordinals)
    assert [p.number for p in parsed.provisions] == ["1", "1", "a", "2"]


# ---------------------------------------------------------------------------
# Footer split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "marker",
    [
        "Nơi nhận:",
        "CHỦ TỊCH QUỐC HỘI",
        "TM. CHÍNH PHỦ",
        "KT. THỦ TƯỚNG",
        "PHỤ LỤC",
        "(*) Nguồn dữ liệu",
    ],
)
def test_footer_markers_end_the_legal_content(marker):
    text = f"Điều 1. Nội dung\n1. Khoản.\n{marker}\nĐiều 99. Không phải điều luật\n"
    main, footer = split_footer(text)

    assert "Điều 1." in main
    assert marker in footer
    assert "Điều 99." in footer


def test_footer_content_never_becomes_a_provision():
    text = "Điều 1. X\nNơi nhận:\nĐiều 99. Trong footer\n"
    parsed = parse_text(text, doc())

    assert [p.number for p in parsed.by_level("article")] == ["1"]
    assert "Điều 99" in parsed.footer


def test_document_without_footer_marker_keeps_everything():
    text = "Điều 1. X\n1. Y.\n"
    main, footer = split_footer(text)

    assert footer == ""
    assert main == text


# ---------------------------------------------------------------------------
# Normalisation and metadata
# ---------------------------------------------------------------------------


def test_text_is_nfc_normalised_before_matching():
    """Decomposed diacritics would break the `Điều` literal in the pattern."""
    decomposed = "Điều 6. Xử phạt\n"  # NFD form of Điều
    parsed = parse_text(decomposed, doc())

    assert len(parsed.by_level("article")) == 1


def test_document_metadata_is_parsed_from_the_source_payload():
    raw = {
        "docIdentity": "168/2024/NĐ-CP",
        "docName": "Nghị định 168",
        "docType": {"docTypeId": 11, "docTypeName": "Nghị định"},
        "effectStatus": {"effectStatusId": 2, "effectStatusName": "Hết Hiệu lực một phần"},
        "issueDate": "2024-12-26T00:00:00",
        "effectDate": "2025-01-01T00:00:00",
        "expireDate": None,
        "fields": [{"fieldId": 1, "fieldName": "Giao thông"}],
        "organs": [{"organId": 1, "organName": "Chính phủ"}],
        "signers": [{"signerId": 1, "signerName": "Trần Hồng Hà"}],
    }
    document = Document.from_metadata("guid", raw)

    assert document.effect_date == date(2025, 1, 1)
    assert document.expire_date is None
    assert document.effect_status == "Hết Hiệu lực một phần"
    assert document.doc_type == "Nghị định"
    assert document.fields == ("Giao thông",)


def test_empty_text_yields_no_provisions():
    parsed = parse_text("", doc())

    assert parsed.provisions == ()
    assert parsed.counts == {"article": 0, "clause": 0, "point": 0}
