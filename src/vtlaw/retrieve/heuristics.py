"""Post-retrieval heuristic re-ranking based on legal amendment status.

After vector + BM25 retrieval and optional cross-encoder rerank, this module
applies two legal-safety adjustments:

1. **Amendment penalty**: if a provision has been ``bãi bỏ`` (abolished) or
   ``thay thế`` (replaced) by a newer document, it should not be cited as
   current law. Penalise it so it sinks below still-in-force provisions.

2. **Recency bonus**: newer documents are more likely to be the current
   applicable law. A small additive boost rewards recent effect dates.

The penalties are applied *after* the eligibility filter, not instead of it.
The filter already removes fully expired documents; this catches provisions
that are still technically ``in_force`` (their parent document has not expired)
but have been individually superseded by an amendment.
"""

from __future__ import annotations

import logging
from datetime import date

from vtlaw.graph.client import GraphClient
from vtlaw.retrieve.search import Hit

log = logging.getLogger(__name__)

# Penalty constants. These are additive to the reranker score and large enough
# to push a hit below any still-in-force provision that the reranker scored
# similarly. The exact values are tunable; the important thing is that
# ``bãi bỏ`` penalises more than ``thay thế``.
ABOLISHED_PENALTY = -5.0
REPLACED_PENALTY = -3.0

# Recency bonus: starts at 2.0 for a document that took effect today, decays
# by 0.3 per year, and floors at 0 (no bonus for very old documents).
RECENCY_INITIAL_BONUS = 2.0
RECENCY_DECAY_PER_YEAR = 0.3


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
    *,
    as_of: date | None = None,
) -> list[Hit]:
    """Apply amendment penalty + recency bonus to the hit list.

    This is a post-processing step after vector search, BM25, and optional
    cross-encoder rerank. It does not reorder drastically — the penalties and
    bonus are small relative to typical reranker scores — but it ensures that
    an abolished provision cannot rank above a still-in-force one when their
    reranker scores are close.

    Args:
        hits: The current ranked hit list.
        client: Neo4j client for amendment lookups.
        as_of: The "current date" for recency calculation. Defaults to today.

    Returns:
        Re-sorted hit list with adjusted scores.
    """
    if not hits:
        return hits

    today = as_of or date.today()
    uids = [h.uid for h in hits]

    # Fetch amendment status for all hits in one query
    abolished_map = fetch_abolished_uids(client, uids)

    # Fetch effect dates for all documents in one query
    doc_identities = list({h.doc_identity for h in hits})
    effect_dates = fetch_doc_effect_dates(client, doc_identities)

    adjusted: list[Hit] = []
    for hit in hits:
        score = hit.score

        # Amendment penalty
        amend_types = abolished_map.get(hit.uid, [])
        if "bãi bỏ" in amend_types:
            score += ABOLISHED_PENALTY
        elif "thay thế" in amend_types:
            score += REPLACED_PENALTY

        # Recency bonus
        eff_str = effect_dates.get(hit.doc_identity)
        if eff_str:
            try:
                eff_date = date.fromisoformat(eff_str)
                years_old = max(0.0, (today - eff_date).days / 365.0)
                recency = max(
                    0.0,
                    RECENCY_INITIAL_BONUS - RECENCY_DECAY_PER_YEAR * years_old,
                )
                score += recency
            except ValueError:
                pass

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
