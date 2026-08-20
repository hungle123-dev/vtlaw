"""Structural patterns for Vietnamese legal text.

Kept in one module so every number/letter rule is visible and testable in one
place rather than scattered through the parser.

Hierarchy, outermost first:

    Phần    (Part)     PHẦN THỨ NHẤT. NHỮNG QUY ĐỊNH CHUNG
    Chương  (Chapter)  Chương I. QUY ĐỊNH CHUNG
    Mục     (Section)  Mục 1. CÁC HÌNH THỨC XỬ PHẠT
    Điều    (Article)  Điều 6. Xử phạt người điều khiển xe ô tô
    Khoản   (Clause)   3. Phạt tiền từ 800.000 đồng đến 1.000.000 đồng
    Điểm    (Point)    a) Điều khiển xe chạy quá tốc độ...

Part, Chapter and Section are optional — decrees routinely skip them.
"""

from __future__ import annotations

import re

# Vietnamese point letters. "đ" sits between "d" and "e", so ordering by ASCII
# codepoint would wrongly place it after "y". There is no f, j, w or z.
POINT_LETTERS = "abcdđeghiklmnopqrstuvxy"

# "đ" goes last inside the character class so it cannot be misread as part of a
# range such as a-d.
_POINT_CLASS = "".join(ch for ch in POINT_LETTERS if ch != "đ") + "đ"

RE_PART = re.compile(r"^(?:PHẦN|Phần)\s+(?:thứ\s+)?(\S+?)\.?\s+(.*)", re.IGNORECASE)
RE_CHAPTER = re.compile(r"^Chương\s+([IVXLCDM]+|\d+)\.?\s*(.*)", re.IGNORECASE)
RE_SECTION = re.compile(r"^Mục\s+(\d+)\.?\s*(.*)", re.IGNORECASE)

# Article and clause numbers may carry a letter suffix: an amending document
# inserts provisions as "Điều 18a" or "khoản 2a" so existing numbering stays
# intact, and a pattern of ^(\d+)\. skips every one of them.
#
# On THIS corpus the suffix branch never fires: all 18 suffixed headers
# (articles 11a, 31b×2, 58e, 58g, 58h, 58i, 58k; clauses 1b, 2b, 3a, 4a, 4b,
# 5a×2, 5b, 5c, 18a) sit inside quoted amendment blocks, which the parser
# treats as content rather than structure — `vtlaw parse check` reports
# letter-suffixed=0. The optional `[a-z]` is kept because the next amending
# document to be added may insert one at the top level, and a silently skipped
# provision is worse than an unused branch.
RE_ARTICLE = re.compile(r"^Điều\s+(\d+[a-z]?)\.?\s*(.*)")

# Clause numbers are bounded to two digits. The highest in the corpus is 75, and
# no line starts with three or more digits followed by a period, so this cannot
# swallow a year ("2024. ...") or a long ordinal list in prose.
RE_CLAUSE = re.compile(r"^(\d{1,2}[a-z]?)\.\s+(.*)")

RE_POINT = re.compile(rf"^([{_POINT_CLASS}])\)\s+(.*)")

# The first match ends the legal content; everything after is signature block,
# distribution list or appendix. Measured: 8,621 characters across the corpus
# (0.5%), containing zero "Điều" headers, so nothing normative is dropped.
RE_FOOTER = re.compile(
    r"^(?:"
    r"Luật này được Quốc hội.*thông qua"
    r"|Nghị định này được Chính phủ.*thông qua"
    r"|CHỦ TỊCH QUỐC HỘI"
    r"|TM\.\s*CHÍNH PHỦ"
    r"|KT\.\s*THỦ TƯỚNG"
    r"|Nơi nhận:"
    r"|\(\*\)\s*Nguồn dữ liệu"
    r"|PHỤ LỤC"
    r"|Phụ lục"
    r")",
    re.MULTILINE,
)

QUOTE_OPEN = ('"', "“")   # " and “
QUOTE_CLOSE = ('"', "”")  # " and ”

# Document identity as it appears in prose: "168/2024/NĐ-CP", "36/2024/QH15".
# One definition, because three modules match it — the citation resolver, the
# graph-template parameter extractor, and the answer citation checker. When they
# drifted apart, a document kind the checker did not recognise made every answer
# citing it read as unsupported.
DOC_IDENTITY = r"\d{1,3}/\d{4}/(?:nđ-cp|qh\d+)"

# Standalone occurrence: not glued to surrounding word characters.
RE_DOC_IDENTITY = re.compile(rf"(?<!\w)(?P<doc>{DOC_IDENTITY})(?!\w)", re.IGNORECASE)


def point_sort_key(letter: str) -> int:
    """Order point letters the way the law does, not the way ASCII does."""
    base = letter.split("#")[0]
    try:
        return POINT_LETTERS.index(base)
    except ValueError:
        return len(POINT_LETTERS)


def opens_quote(line: str) -> bool:
    return line[:1] in QUOTE_OPEN


def closes_quote(line: str) -> bool:
    """True when a line ends a quoted block.

    Trailing punctuation is stripped first: quoted amendments usually end as
    `..."` or `...".;` where the closing mark is not the final character.
    """
    return line.rstrip(".;:, ")[-1:] in QUOTE_CLOSE
