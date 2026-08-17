"""Post-retrieval demotion of provisions that an amendment has superseded.

If a provision has been ``bãi bỏ`` (abolished) or ``thay thế`` (replaced) by a
newer document it should not be cited as current law, so it is pushed below
still-in-force provisions.

This runs *after* the eligibility filter, not instead of it. The filter already
removes fully expired documents; this catches provisions whose parent document is
still in force but which have been individually superseded.

There is no recency bonus. One existed — a positive score for documents with a
recent ``effect_date`` — and it was removed after measuring it on two datasets
with opposite labelling:

    QA_Part2345 (labelled against 168/2024, 36/2024)   MRR 0.528 -> 0.593  (+0.066)
    QA_NLP      (labelled against 100/2019)            MRR 0.568 -> 0.433  (-0.135)

Weighted across both: +0.0015, i.e. nothing. The bonus was not detecting the
applicable law, it was detecting which dataset was being scored, and it lost more
on the older-law set than it gained on the newer one.

It could not be rescued by tuning either. Being additive, it applied to every hit
regardless of what retrieval thought, and at any strength large enough to matter
it saturated into "sort by effect date" — measured: at 3x the score span, 200 of
200 questions had an identical top ten to 1x, the retrieval signal fully erased.
Rescaling the constants from absolute values to a fraction of the score span only
moved where that saturation began.

"Which law applies" is a filter on a date, not a score adjustment. The graph
carries ``effect_date`` and ``expire_date`` for exactly that, and the retrieval
queries already filter on them.
"""

from __future__ import annotations

import logging

from vtlaw.graph.client import GraphClient
from vtlaw.retrieve.search import Hit

log = logging.getLogger(__name__)

# Expressed as a FRACTION of the score span of the list being reranked, so the
# demotion means the same thing whether it is applied to RRF scores (span ~0.3)
# or cross-encoder scores (span ~10). Absolute constants here were a bug: -5.0
# against an RRF list is 18x the whole span, which does not demote a hit, it
# replaces the ranking.
#
# Both are larger than 1.0 span deliberately: an amendment is a fact recorded in
# the graph, not a guess, so a superseded provision should fall below every
# in-force one rather than merely lose a few places.
ABOLISHED_PENALTY = -2.0
REPLACED_PENALTY = -1.2

# A degenerate list (every hit scored identically) has no span to scale by. Fall
# back to this so an abolished provision is still demoted rather than left alone.
_FALLBACK_SPAN = 1.0


def fetch_amend_status(
    client: GraphClient,
    uids: list[str],
) -> dict[str, list[str]]:
    """Fetch amendment types for the given UIDs.

    Returns a mapping ``uid -> ["bãi bỏ", "thay thế", ...]``. A UID with no
    incoming AMENDS edges gets an empty list.

    An amendment can target a provision at any level: an Article, a Clause, or
    a Point. The query MATCHes the target by UID directly, so it works
    regardless of label.
    """
    if not uids:
        return {}

    with client.session() as session:
        rows = session.run(
            """
            UNWIND $uids AS target_uid
            MATCH (target)-[:AMENDS {type: $amend_type}]-(source)
            RETURN target_uid AS uid, collect(DISTINCT source) AS sources
            """,
            uids=uids,
            amend_type=None,
        ).data()

    # The query above doesn't filter by type; we need to collect types.
    # Let's do a simpler query that gets the types directly.
    with client.session() as session:
        rows = session.run(
            """
            UNWIND $uids AS target_uid
            MATCH (source)-[r:AMENDS]->(target {uid: target_uid})
            RETURN target_uid AS uid, collect(DISTINCT r.type) AS types
            """,
            uids=uids,
        ).data()

    return {row["uid"]: row["types"] for row in rows}


def fetch_abolished_uids(
    client: GraphClient,
    uids: list[str],
) -> dict[str, list[str]]:
    """Check which UIDs have been abolished or replaced.

    Returns a mapping ``uid -> ["bãi bỏ", "thay thế"]`` for UIDs that have
    at least one such amendment. UIDs with no abolishment are absent from
    the result.
    """
    if not uids:
        return {}

    with client.session() as session:
        rows = session.run(
            """
            UNWIND $uids AS target_uid
            MATCH (source)-[r:AMENDS]->(target {uid: target_uid})
            WHERE r.type IN ['bãi bỏ', 'thay thế']
            RETURN target_uid AS uid, collect(DISTINCT r.type) AS types
            """,
            uids=uids,
        ).data()

    return {row["uid"]: row["types"] for row in rows}


def fetch_doc_effect_dates(
    client: GraphClient,
    doc_identities: list[str],
) -> dict[str, str | None]:
    """Get effect_date for documents by doc_identity.

    Returns a mapping ``doc_identity -> "YYYY-MM-DD" or None``.
    """
    if not doc_identities:
        return {}

    with client.session() as session:
        rows = session.run(
            """
            UNWIND $doc_ids AS did
            MATCH (d:Document {doc_identity: did})
            RETURN d.doc_identity AS doc_id,
                   CASE WHEN d.effect_date IS NULL THEN NULL
                        ELSE toString(d.effect_date) END AS effect_date
            """,
            doc_ids=doc_identities,
        ).data()

    return {row["doc_id"]: row["effect_date"] for row in rows}


def apply_heuristic_rerank(
    hits: list[Hit],
    client: GraphClient,
) -> list[Hit]:
    """Demote provisions an amendment has abolished or replaced.

    The demotion is a fraction of the list's own score span, so it means the
    same on any score scale. It is deliberately strong: an AMENDS edge is a
    recorded fact, so a superseded provision belongs below the in-force ones
    rather than a few places lower.

    Args:
        hits: The current ranked hit list.
        client: Neo4j client for amendment lookups.

    Returns:
        Re-sorted hit list with adjusted scores.
    """
    if not hits:
        return hits

    uids = [h.uid for h in hits]
    abolished_map = fetch_abolished_uids(client, uids)
    if not abolished_map:
        # Nothing in this list is superseded, so there is nothing to reorder.
        return hits

    scores = [h.score for h in hits]
    span = max(scores) - min(scores)
    if span <= 0:
        span = _FALLBACK_SPAN

    adjusted: list[Hit] = []
    for hit in hits:
        score = hit.score
        amend_types = abolished_map.get(hit.uid, [])
        if "bãi bỏ" in amend_types:
            score += ABOLISHED_PENALTY * span
        elif "thay thế" in amend_types:
            score += REPLACED_PENALTY * span

        adjusted.append(
            Hit(
                uid=hit.uid,
                score=score,
                doc_identity=hit.doc_identity,
                label=hit.label,
                content=hit.content,
                title=hit.title,
            )
        )

    adjusted.sort(key=lambda h: h.score, reverse=True)
    return adjusted
