"""Post-retrieval demotion of provisions that an amendment has superseded.

If a provision has been ``bãi bỏ`` (abolished), ``thay thế`` (replaced),
``sửa đổi`` (modified), or ``sửa đổi, bổ sung`` by a newer document it should
not be cited as current law, so it is pushed below still-in-force provisions.

This runs *after* the eligibility filter, not instead of it. The filter already
removes fully expired documents; this catches provisions whose parent document is
still in force but which have been individually superseded.

Which law applies is first a date filter, not a recency bonus. The graph carries
``effect_date`` and ``expire_date`` for that filter; this module only handles the
more specific amendment annotations after retrieval.
"""

from __future__ import annotations

from datetime import date

from vtlaw.graph.client import GraphClient
from vtlaw.retrieve.search import Hit

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


def fetch_abolished_uids(
    client: GraphClient,
    uids: list[str],
    *,
    as_of: date | None = None,
) -> dict[str, list[str]]:
    """Check which UIDs have been superseded by an effective amendment.

    Returns a mapping ``uid -> [amend_type, ...]`` for UIDs that have at least
    one repeal, replacement, or modifying amendment. ``bổ sung`` alone is not
    included because it does not supersede the target provision. An amendment
    aimed at an Article or Clause also applies to its retrieved descendants.
    """
    if not uids:
        return {}

    cutoff = (as_of or date.today()).isoformat()
    with client.session() as session:
        rows = session.run(
            """
            UNWIND $uids AS target_uid
            MATCH (target {uid: target_uid})
            MATCH (source)-[r:AMENDS]->(amended)
            MATCH (source_document:Document {doc_identity: source.doc_identity})
            WHERE r.type IN ['bãi bỏ', 'thay thế', 'sửa đổi', 'sửa đổi, bổ sung']
              AND source_document.effect_date <= date($as_of)
              AND (
                  amended.uid = target.uid
                  OR EXISTS {
                      MATCH (amended:Article)-[:HAS_CLAUSE]->(target:Clause)
                  }
                  OR EXISTS {
                      MATCH (amended:Article)-[:HAS_CLAUSE]->(:Clause)-[:HAS_POINT]->(target:Point)
                  }
                  OR EXISTS {
                      MATCH (amended:Clause)-[:HAS_POINT]->(target:Point)
                  }
              )
            RETURN target_uid AS uid, collect(DISTINCT r.type) AS types
            """,
            uids=uids,
            as_of=cutoff,
        ).data()

    return {row["uid"]: row["types"] for row in rows}


def apply_heuristic_rerank(
    hits: list[Hit],
    client: GraphClient,
    *,
    as_of: date | None = None,
) -> list[Hit]:
    """Demote provisions an amendment has superseded.

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
    abolished_map = fetch_abolished_uids(client, uids, as_of=as_of)
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
        elif (
            "thay thế" in amend_types
            or "sửa đổi" in amend_types
            or "sửa đổi, bổ sung" in amend_types
        ):
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
