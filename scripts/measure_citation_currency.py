"""Do "độ hiện hành" của trích dẫn: câu trả lời có dẫn văn bản mới nhất trong evidence?

Không phải Recall@k. Recall đo retrieval; lỗi ở đây là generation: cả 100/2019
và 168/2024 đều nằm trong evidence, mô hình chọn cái cũ. SYSTEM_PROMPT quy tắc 5
đã yêu cầu ưu tiên văn bản hiệu lực gần nhất, nên phép đo này chỉ hỏi: generation
có tuân quy tắc của chính nó không?

Chạy:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/measure_citation_currency.py [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

from vtlaw.config import get_settings
from vtlaw.embed import Embedder
from vtlaw.generate.answer import AnswerGenerator
from vtlaw.generate.llm_client import LLMClient
from vtlaw.graph.client import GraphClient
from vtlaw.parse.patterns import RE_DOC_IDENTITY
from vtlaw.retrieve.search import HybridRetriever

AS_OF = date(2026, 8, 20)


def _effect_dates(client: GraphClient) -> dict[str, date]:
    with client.session() as session:
        rows = session.run(
            "MATCH (d:Document) RETURN d.doc_identity AS ident, d.effect_date AS eff"
        ).data()
    return {r["ident"]: r["eff"].to_native() for r in rows if r["eff"]}


def _cited_documents(answer: str) -> set[str]:
    return {m.group("doc").upper() for m in RE_DOC_IDENTITY.finditer(answer)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--dataset", default="data/evaluation/qa/QA_NLP.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    settings = get_settings()
    client = GraphClient(settings)
    embedder = Embedder(settings)
    retriever = HybridRetriever(client, embedder, settings)
    generator = AnswerGenerator(client, embedder, LLMClient(settings), settings)
    effect = _effect_dates(client)

    with open(args.dataset, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["question"].strip()][: args.limit]

    # Only questions whose evidence spans more than one decree can be stale:
    # with a single source document there is nothing older to pick.
    contested = 0
    stale = 0
    detail = []

    for i, row in enumerate(rows, 1):
        question = row["question"].strip()
        result = retriever.search_and_rerank(
            question, k=settings.context_k, strategy="hybrid",
            heuristic_rerank=True, as_of=AS_OF,
        )
        if not result.hits:
            continue

        # Only penalty decrees compete; a Luật and a Nghị định are different
        # instruments, not two versions of one rule.
        decrees = {h.doc_identity for h in result.hits if h.doc_identity.endswith("NĐ-CP")}
        if len(decrees) < 2:
            continue
        contested += 1
        newest = max(decrees, key=lambda d: effect.get(d, date.min))

        answer = generator.generate_from_hits(question, result.hits, as_of=AS_OF)
        cited = _cited_documents(answer) & decrees
        cited_stale = {d for d in cited if effect.get(d, date.min) < effect[newest]}
        # A question that names a decree ("Nghị định 100/2019/NĐ-CP quy định gì?")
        # is asking about that document. Citing it back is correct, not stale.
        asked_for = _cited_documents(question) & decrees
        is_stale = bool(cited_stale - asked_for) and newest not in cited

        if is_stale:
            stale += 1
        detail.append({
            "question": question,
            "decrees_in_evidence": sorted(decrees),
            "newest_available": newest,
            "cited": sorted(cited),
            "stale": is_stale,
        })
        flag = "STALE" if is_stale else "ok   "
        print(f"[{i:3}/{len(rows)}] {flag} newest={newest:16} cited={sorted(cited)}")

    rate = stale / contested if contested else 0.0
    print()
    print(f"contested questions (evidence spans 2+ decrees): {contested}")
    print(f"answers citing only a superseded decree        : {stale}")
    print(f"stale-citation rate                           : {rate:.3f}")

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "as_of": AS_OF.isoformat(),
                    "dataset": args.dataset,
                    "rows_scored": len(rows),
                    "contested": contested,
                    "stale": stale,
                    "stale_rate": rate,
                    "detail": detail,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"written to {args.output}")

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
