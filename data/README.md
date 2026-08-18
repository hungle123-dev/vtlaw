# Data

Every artifact the pipeline needs, committed so the system is reproducible
**without network access**.

## Provenance

| | |
|---|---|
| Source | `phapluat.gov.vn` — JSON API (`/api/legal-documents`) |
| Collected | 2026-03 – 2026-04 |
| Collected by | `n3sfan/NLP-LegalQA` (MIT, © Le Truong Thinh), used with permission |
| Vendored here | 2026-08-15 |

The source API is a public government service with no SLA. As of 2026-08-15 it is
unavailable: search returns `HTTP 200` with `error: "QTDC API error: 500"`, and
after repeated probing the host answered `403` from a `PWS/8.3.1.0.8` WAF. That is
precisely why the corpus is committed rather than fetched on demand.

`phapluat.gov.vn` is the only source that serves **structured text**. The
alternatives were probed and rejected:

| Source | Result |
|---|---|
| `vanban.chinhphu.vn` | HTML carries metadata only; the document is a signed PDF whose `extract_text()` returns 0 characters |
| `congbao.chinhphu.vn` | Official gazette, also scanned PDF |
| `vbpl.vn` | `403` + reCAPTCHA |
| `thuvienphapluat.vn` | `403` |

OCR is not used: one misread digit changes a penalty amount.

## Layout

```
data/
├── snapshot/                      immutable, hash-verified copy of the source
│   ├── documents/{doc_guid}.txt   12 files, 31K–322K — plain text, NFC
│   ├── metadata/{doc_guid}.json   12 files — the `tomtat` payload
│   └── manifest.json              per-document hashes + provenance
├── amends/{doc_guid}.json         3 amendment-relation annotation files
├── annotations/EXTRACTION_PROMPT.txt
│                                  prompt used to produce the amendment labels
└── evaluation/
    └── qa/QA_*.csv                7 files — question / answer / reference
```

Files are keyed by `docGUId` (a UUID), not `docIdentity`: values such as
`100/2019/NĐ-CP` are not unique across source revisions and would collide on disk.

## Corpus

12 documents, 1,645,697 characters of text, parsing to **7,381 provisions**
(566 articles, 2,763 clauses, 4,052 points).

| Document | Effect date | Status |
|---|---|---|
| `15/2012/QH13` | 2013-07-01 | Hết Hiệu lực một phần |
| `44/2019/QH14` | 2020-01-01 | Hết Hiệu lực một phần |
| `100/2019/NĐ-CP` | 2020-01-01 | Hết Hiệu lực một phần |
| `67/2020/QH14` | 2022-01-01 | Hết Hiệu lực một phần |
| `123/2021/NĐ-CP` | 2022-01-01 | Hết Hiệu lực một phần |
| `119/2024/NĐ-CP` | 2024-10-01 | Còn Hiệu lực |
| `35/2024/QH15` | 2025-01-01 | Hết Hiệu lực một phần |
| `36/2024/QH15` | 2025-01-01 | Còn Hiệu lực |
| `56/2024/QH15` | 2025-01-01 | Hết Hiệu lực một phần |
| `168/2024/NĐ-CP` | 2025-01-01 | Hết Hiệu lực một phần |
| `88/2025/QH15` | 2025-07-01 | Còn Hiệu lực |
| `118/2025/QH15` | 2026-07-01 | Chưa áp dụng |

Holding both `100/2019/NĐ-CP` and `168/2024/NĐ-CP` matters: the same offence
carries different penalties depending on when it happened, so the corpus can
exercise temporal reasoning rather than only topical matching.

## Evaluation labels

The QA files are NLP-LegalQA labels, not a new independent benchmark assembled
by this project. `QA_Part2`, `QA_Part3`, `QA_Part4`, and `QA_Part5` are subsets
of the larger `QA_Part234` / `QA_Part2345` files; pooling or averaging all seven
would count many questions more than once. Report `QA_NLP.csv` (94 rows) and
`QA_Part2345.csv` (200 rows) as two separate tracks.

The rows do not provide a date-of-fact. A retrieval score therefore measures
agreement with the supplied historical UID label, not whether the provision is
the legally current rule after an amendment. Every reproducible run records its
`as_of` date, dataset SHA-256, models, and retrieval settings in its JSON output.

## Known defects in the source data

Measured, not assumed. Each has a test.

**1. Duplicate provision numbering.** `118/2025/QH15` Điều 5 contains two
distinct `khoản 4` and two distinct `khoản 5`, each amending a different law:

```
khoản 4  "Sửa đổi, bổ sung khoản 4 Điều 25 như sau: …"
khoản 4  "Sửa đổi, bổ sung điểm a khoản 1 Điều 29 như sau: …"
```

A UID of `(doc, article, clause)` merges them. Measured on that document: 123
clause rows collapse to 121 distinct UIDs — two provisions of legal text lost. A
`#n` suffix on the UID keeps both; the citation still renders as
`Khoản 4 Điều 5 118/2025/QH15`.

**2. Envelope errors under HTTP 200.** Search returns
`{"data": {"rowCount": 0, "docs": []}, "error": "QTDC API error: 500"}` with
status 200. `raise_for_status()` passes, so a caller checking only the status
reads an upstream failure as "end of catalog" and stops paginating with a
complete-looking empty result.

**3. Letter-suffixed numbering is *not* a defect.** Amending documents insert
provisions as `khoản 2a` or `Điều 18a`. An earlier measurement of mine suggested
a `^(\d+)\.` clause pattern loses 17 of these. Re-measuring with quote tracking
shows all 18 such lines sit **inside quoted amendment text** — they are provisions
being inserted into *other* laws, not headings of the document being parsed.
Skipping them is correct. `tests/parse/test_corpus.py` pins this so a future "fix"
cannot fabricate provisions.

## Reproducing

```bash
docker compose up -d                  # Neo4j on 127.0.0.1:17687

vtlaw scrape verify                   # offline: hash-check the snapshot
vtlaw scrape status                   # what the snapshot holds
vtlaw parse check                     # offline: parse and report counts
vtlaw graph import --wipe             # snapshot -> Neo4j
vtlaw graph status                    # counts read back from the database
```

Only `vtlaw scrape run` needs the network. It verifies TLS, sets timeouts, retries
transient failures with bounded jittered backoff, throttles between requests, and
refuses to treat an envelope error as an empty catalog. It stops immediately on
`403` rather than retrying a refusal.

## Licence

Corpus text is Vietnamese legal material published by the state. The tooling that
produced this snapshot is MIT-licensed (© Le Truong Thinh) and used with the
author's permission.
