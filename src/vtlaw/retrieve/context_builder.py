"""Graph-enhanced context building for retrieved provisions.

A provision on its own is not an answer: a Point states the offence, the penalty
amount lives in its parent Clause, and the Article title gives the topic. This
module walks the graph hierarchy to assemble full context for each hit.

Three strategies, each filling a gap the others leave:

1. **fetch_hierarchy**: walks UP from a hit to its Document, assembling the
   full citation path (Điều → Khoản → Điểm) with content at each level.

2. **fetch_sibling_points**: walks SIDEWAYS from a Point to its siblings under
   the same Clause. A Point often references "the cases listed below" — without
   the siblings, that reference is dangling.

3. **fetch_children_context**: walks DOWN from an Article or Clause to its
   descendants. An Article's own content is often just a title; its Clauses
   carry the actual penalties.
"""

from __future__ import annotations

from datetime import date

from vtlaw.graph.client import GraphClient
from vtlaw.retrieve.search import Hit

# ---------------------------------------------------------------------------
# Fetch hierarchy (walk UP)
# ---------------------------------------------------------------------------

_HIERARCHY_QUERY = """
UNWIND $uids AS target_uid
MATCH path = (d:Document)
      -[:HAS_ARTICLE|HAS_CLAUSE|HAS_POINT*]->(target)
WHERE target.uid = target_uid
RETURN target.uid AS uid,
       [n IN nodes(path)[1..] | {labels: labels(n), props: properties(n)}] AS hierarchy,
       d.doc_identity AS doc_identity,
       toString(d.effect_date) AS effect_date
"""

_LABEL_VI = {
    "Part": "Phần",
    "Chapter": "Chương",
    "Section": "Mục",
    "Article": "Điều",
    "Clause": "Khoản",
    "Point": "Điểm",
}


def _retain_rendered_hit(
    rendered_evidence: dict[str, Hit] | None,
    node: dict,
    label: str,
    doc_identity: str,
) -> None:
    """Keep the full provision record for text added to the prompt."""
    if (
        rendered_evidence is None
        or label not in ("Article", "Clause", "Point")
        or not (uid := node.get("uid"))
    ):
        return
    rendered_evidence[str(uid)] = Hit(
        uid=str(uid),
        score=0.0,
        doc_identity=str(node.get("doc_identity") or doc_identity),
        label=label,  # type: ignore[arg-type]
        content=str(node.get("content") or ""),
        title=node.get("title"),
    )


def fetch_hierarchy(
    client: GraphClient,
    uids: list[str],
    *,
    evidence_uids: set[str] | None = None,
    rendered_evidence: dict[str, Hit] | None = None,
) -> dict[str, str]:
    """Walk UP from each hit to its Document, building a citation context.

    Returns a mapping ``uid -> formatted context string``. The string includes
    the document header (identity + effect date), the hierarchy path
    (Chương > Mục > Điều > Khoản > Điểm) with titles, and the target's own
    content.
    """
    if not uids:
        return {}

    result: dict[str, str] = {}

    with client.session() as session:
        records = session.run(_HIERARCHY_QUERY, uids=uids).data()

    for record in records:
        uid = record["uid"]
        entries = record["hierarchy"] or []

        lines: list[str] = []
        for index, entry in enumerate(entries):
            # The query projects {labels, props} explicitly. Reading `labels(n)`
            # server-side is the only way to get them: `nodes(path)` through
            # Result.data() arrives as bare property dicts, so every label test
            # fell through and the whole hierarchy — including the parent Clause
            # that carries the penalty amount — was dropped from the context.
            labels = entry.get("labels") or []
            node = entry.get("props") or {}
            label = labels[0] if labels else ""
            is_target = index == len(entries) - 1
            rendered = False

            if label == "Article":
                title = node.get("title")
                number = node.get("number", "")
                lines.append(
                    f"Điều {number}: {title}" if title else f"Điều {number}"
                )
                rendered = True
            elif label == "Clause":
                number = node.get("number", "")
                content = (node.get("content") or "").strip()
                # The target's own content is appended below as "Nội dung"; an
                # ancestor Clause contributes the penalty the Point refers to.
                if is_target:
                    lines.append(f"Khoản {number}.")
                elif content:
                    lines.append(f"Khoản {number}.\n{content}")
                else:
                    lines.append(f"Khoản {number}.")
                rendered = True
            elif label == "Point":
                letter = node.get("letter", "")
                lines.append(f"Điểm {letter}.")
                rendered = True
            elif label in _LABEL_VI:
                number = node.get("number", "")
                title = node.get("title")
                vn_label = _LABEL_VI[label]
                lines.append(
                    f"{vn_label} {number}: {title}" if title else f"{vn_label} {number}"
                )
                rendered = True

            if rendered and evidence_uids is not None and (node_uid := node.get("uid")):
                evidence_uids.add(str(node_uid))
            if rendered:
                _retain_rendered_hit(
                    rendered_evidence,
                    node,
                    label,
                    str(record["doc_identity"] or ""),
                )

        # Target's own content
        target = (entries[-1].get("props") or {}) if entries else {}
        main_content = (target.get("content") or "").strip()

        # Document header
        doc_id = record["doc_identity"] or ""
        eff_str = record["effect_date"] or "N/A"
        header = f"[Văn bản: {doc_id} — Hiệu lực: {eff_str}]"

        body = "\n".join(lines)
        if main_content:
            body += f"\nNội dung: {main_content}"

        result[uid] = f"{header}\n{body}".strip()

    return result


# ---------------------------------------------------------------------------
# Fetch sibling points (walk SIDEWAYS)
# ---------------------------------------------------------------------------

_SIBLING_QUERY = """
UNWIND $uids AS target_uid
MATCH (target:Point {uid: target_uid})
MATCH (clause:Clause)-[:HAS_POINT]->(target)
MATCH (clause)-[:HAS_POINT]->(sibling:Point)
WHERE sibling.uid <> target_uid
RETURN target.uid AS uid,
       collect({uid: sibling.uid, letter: sibling.letter, content: sibling.content}) AS siblings
"""


def fetch_sibling_points(
    client: GraphClient,
    uids: list[str],
    *,
    evidence_uids: set[str] | None = None,
    rendered_evidence: dict[str, Hit] | None = None,
) -> dict[str, str]:
    """Fetch sibling Points under the same Clause.

    A Point often references "the cases listed below" — without siblings, that
    reference is dangling. Returns ``uid -> formatted text`` for Points that
    have siblings.
    """
    if not uids:
        return {}

    result: dict[str, str] = {}

    with client.session() as session:
        records = session.run(_SIBLING_QUERY, uids=uids).data()

    for record in records:
        siblings = record["siblings"] or []
        if not siblings:
            continue

        siblings.sort(key=lambda s: s.get("letter", ""))
        lines = []
        for s in siblings:
            letter = s.get("letter", "?")
            content = (s.get("content") or "").strip()
            if content:
                lines.append(f"  Điểm {letter}. {content}")
                if evidence_uids is not None and (sibling_uid := s.get("uid")):
                    evidence_uids.add(str(sibling_uid))
                _retain_rendered_hit(
                    rendered_evidence,
                    s,
                    "Point",
                    str(s.get("doc_identity") or str(s.get("uid") or "").split("::", 1)[0]),
                )

        if lines:
            result[record["uid"]] = "\n".join(lines)

    return result


# ---------------------------------------------------------------------------
# Fetch children context (walk DOWN)
# ---------------------------------------------------------------------------

_CHILDREN_QUERY = """
UNWIND $uids AS target_uid
MATCH (target) WHERE target.uid = target_uid
OPTIONAL MATCH (target)-[:HAS_CLAUSE|HAS_POINT*1..2]->(child)
WITH target.uid AS uid,
     labels(target)[0] AS target_label,
     collect({
        label: CASE WHEN child IS NULL THEN NULL ELSE labels(child)[0] END,
        number: child.number,
        letter: child.letter,
        content: child.content,
        uid: child.uid
     }) AS children
RETURN uid, target_label, children
"""


def fetch_children_context(
    client: GraphClient,
    uids: list[str],
    *,
    evidence_uids: set[str] | None = None,
    rendered_evidence: dict[str, Hit] | None = None,
) -> dict[str, str]:
    """Fetch descendant content for Article/Clause nodes.

    An Article's own content is often just a title; its Clauses carry the
    actual penalties. Returns ``uid -> formatted text`` for nodes with children.
    """
    if not uids:
        return {}

    result: dict[str, str] = {}

    with client.session() as session:
        records = session.run(_CHILDREN_QUERY, uids=uids).data()

    for record in records:
        label = record["target_label"]
        if label not in ("Article", "Clause"):
            continue

        children = [c for c in (record["children"] or []) if c.get("content")]
        if not children:
            continue

        # Sort by uid to maintain document order
        children.sort(key=lambda c: c.get("uid", ""))

        lines = []
        for child in children:
            child_label = child.get("label", "")
            content = (child.get("content") or "").strip()
            if not content:
                continue

            if child_label == "Clause":
                num = child.get("number", "?")
                lines.append(f"Khoản {num}. {content}")
            elif child_label == "Point":
                letter = child.get("letter", "?")
                lines.append(f"  Điểm {letter}. {content}")
            else:
                continue
            if evidence_uids is not None and (child_uid := child.get("uid")):
                evidence_uids.add(str(child_uid))
            _retain_rendered_hit(
                rendered_evidence,
                child,
                child_label,
                str(
                    child.get("doc_identity")
                    or str(child.get("uid") or "").split("::", 1)[0]
                ),
            )

        if lines:
            result[record["uid"]] = "\n".join(lines)

    return result


# ---------------------------------------------------------------------------
# Build full context
# ---------------------------------------------------------------------------


def build_full_context(
    client: GraphClient,
    hits: list[Hit],
    *,
    as_of: date | None = None,
    evidence_uids: set[str] | None = None,
    rendered_evidence: dict[str, Hit] | None = None,
) -> dict[str, str]:
    """Assemble complete context for each hit using all three strategies.

    Returns a mapping ``uid -> enriched context text`` that combines:
    - The hierarchy path (from fetch_hierarchy)
    - Sibling points (from fetch_sibling_points, for Point hits)
    - Children context (from fetch_children_context, for Article/Clause hits)
    """
    if not hits:
        return {}

    uids = [h.uid for h in hits]
    # Walk UP: get hierarchy for all hits
    hierarchy = fetch_hierarchy(
        client,
        uids,
        evidence_uids=evidence_uids,
        rendered_evidence=rendered_evidence,
    )

    # Walk SIDEWAYS: get siblings for Point hits
    point_uids = [h.uid for h in hits if h.label == "Point"]
    siblings = fetch_sibling_points(
        client,
        point_uids,
        evidence_uids=evidence_uids,
        rendered_evidence=rendered_evidence,
    )

    # Walk DOWN: get children for Article/Clause hits
    parent_uids = [h.uid for h in hits if h.label in ("Article", "Clause")]
    children = fetch_children_context(
        client,
        parent_uids,
        evidence_uids=evidence_uids,
        rendered_evidence=rendered_evidence,
    )

    # Combine
    result: dict[str, str] = {}
    for hit in hits:
        parts = []

        # Start with hierarchy (includes document header + path + content)
        ctx = hierarchy.get(hit.uid)
        if ctx:
            parts.append(ctx)

        # Add siblings for Point hits
        sibling_text = siblings.get(hit.uid)
        if sibling_text:
            parts.append(f"Các điểm khác trong cùng khoản:\n{sibling_text}")

        # Add children for Article/Clause hits
        child_text = children.get(hit.uid)
        if child_text:
            parts.append(f"Nội dung chi tiết:\n{child_text}")

        if parts:
            result[hit.uid] = "\n\n".join(parts)
        else:
            result[hit.uid] = hit.content
            if evidence_uids is not None:
                evidence_uids.add(hit.uid)
            if rendered_evidence is not None:
                rendered_evidence[hit.uid] = hit

    return result
