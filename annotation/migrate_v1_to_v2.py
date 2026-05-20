"""One-shot migration of annotation/to_annotate.json from schema v1 to v2.

Per the spec (docs/superpowers/specs/2026-05-20-schema-v2-typed-node-extension-design.md
§6), the rewrite is by (question_id, db_id, node_id) -- we never pattern-match
on the free-text 'definition' field. ValueMap siblings are never deleted.

Idempotent: running on an already-migrated item is a no-op.

Usage:
  .venv/bin/python annotation/migrate_v1_to_v2.py annotation/to_annotate.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# ----- v2 migration rules, keyed by (qid, db_id) ---------------------------

# Pattern A: replace one Concept-with-definition node with a single-binding ColumnAlias.
# Spec §6 row 1.
COLUMN_ALIAS_SINGLE = {
    # (qid, db_id, concept_id): {"name": ..., "table": ..., "column": ...}
    (491, "card_games", "c1"): {
        "name": "magic card market name", "table": "sets", "column": "mcmName"},
    (1530, "debit_card_specializing", "c1"): {
        "name": "full product name", "table": "products", "column": "Description"},
    (776, "superhero", "c1"): {
        "name": "hero name", "table": "superhero", "column": "superhero_name"},
    (1052, "european_football_2", "c1"): {
        "name": "preferred foot in attacking",
        "table": "Player_Attributes", "column": "preferred_foot"},
    (122, "financial", "c1"): {
        "name": "district name", "table": "district", "column": "A2"},
    (122, "financial", "c2"): {
        "name": "region name", "table": "district", "column": "A3"},
    (769, "superhero", "c1"): {
        "name": "which superhero (identity column)",
        "table": "superhero", "column": "superhero_name"},
}

# Pattern B: replace one Concept-with-definition node with a multi-binding ColumnAlias.
# Spec §6 row 2.
COLUMN_ALIAS_MULTI = {
    (978, "formula_1", "c1"): {
        "name": "location coordinates",
        "bindings": [{"table": "circuits", "column": "lat"},
                     {"table": "circuits", "column": "lng"}]},
    (1364, "student_club", "c1"): {
        "name": "hometown",
        "bindings": [{"table": "zip_code", "column": "city"},
                     {"table": "zip_code", "column": "county"},
                     {"table": "zip_code", "column": "state"}]},
}

# Pattern C: add grounding entries with 'value' to a Formula. Spec §6 row 3.
# (qid, db_id, formula_id): list of {term, table, column, value} entries to add.
# We APPEND -- never replace -- so existing bare-column grounding stays intact.
GROUNDING_VALUE_ADDITIONS = {
    (65, "california_schools", "f1"): [
        {"term": "locally funded", "table": "schools",
         "column": "FundingType", "value": "Locally funded"},
        {"term": "Santa Clara", "table": "schools",
         "column": "County", "value": "Santa Clara"},
    ],
    (634, "codebase_community", "f1"): [
        {"term": "Harvey Motulsky", "table": "users",
         "column": "DisplayName", "value": "Harvey Motulsky"},
        {"term": "Noah Snyder", "table": "users",
         "column": "DisplayName", "value": "Noah Snyder"},
    ],
    (769, "superhero", "f1"): [
        {"term": "durability", "table": "attribute",
         "column": "attribute_name", "value": "durability"},
    ],
    (786, "superhero", "f1"): [
        {"term": "Strength", "table": "attribute",
         "column": "attribute_name", "value": "Strength"},
    ],
    (317, "toxicology", "f1"): [
        {"term": "+", "table": "molecule", "column": "label", "value": "+"},
        {"term": "cl", "table": "atom", "column": "element", "value": "cl"},
    ],
}

# Pattern D: add depends_on to a Formula. Spec §6 row 4.
# (qid, db_id, formula_id): list of ids in the same record that this formula depends on.
DEPENDS_ON_ADDITIONS = {
    (1482, "debit_card_specializing", "f1"): ["r1"],
    (1482, "debit_card_specializing", "f2"): ["r1"],
    (954, "formula_1", "f1"): ["r1", "r2"],
}


# ----- migration core ------------------------------------------------------


def migrate_item(item: dict) -> dict:
    """Apply v2 rewrites in place on a single record, return the same item.

    Idempotent: if a node has already been rewritten (e.g. is already a
    ColumnAlias, or its Formula already has the relevant grounding entry),
    that step is a no-op.
    """
    qid = item.get("question_id")
    db = item.get("db_id")

    new_nodes = []
    for node in item.get("nodes", []):
        nid = node.get("id")
        key = (qid, db, nid)

        # Pattern A: Concept-with-definition -> ColumnAlias single binding.
        if key in COLUMN_ALIAS_SINGLE and node.get("type") == "Concept":
            r = COLUMN_ALIAS_SINGLE[key]
            new_nodes.append({
                "id": nid,
                "type": "ColumnAlias",
                "name": r["name"],
                "bindings": [{"table": r["table"], "column": r["column"]}],
            })
            continue

        # Pattern B: Concept-with-definition -> ColumnAlias multi binding.
        if key in COLUMN_ALIAS_MULTI and node.get("type") == "Concept":
            r = COLUMN_ALIAS_MULTI[key]
            new_nodes.append({
                "id": nid,
                "type": "ColumnAlias",
                "name": r["name"],
                "bindings": list(r["bindings"]),
            })
            continue

        # Pattern C + D apply to Formula nodes; rewrite below after appending.
        new_nodes.append(node)

    item["nodes"] = new_nodes

    # Pattern C: add grounding-with-value entries to Formula nodes.
    for node in item["nodes"]:
        if node.get("type") != "Formula":
            continue
        key = (qid, db, node.get("id"))
        adds = GROUNDING_VALUE_ADDITIONS.get(key)
        if not adds:
            continue
        existing = node.setdefault("grounding", [])
        for new_g in adds:
            # idempotent: skip if an entry with the same (term, table, column, value) is present
            present = any(
                g.get("term") == new_g["term"]
                and g.get("table") == new_g["table"]
                and g.get("column") == new_g["column"]
                and g.get("value") == new_g["value"]
                for g in existing
            )
            if not present:
                existing.append(dict(new_g))

    # Pattern D: add depends_on to Formula nodes.
    for node in item["nodes"]:
        if node.get("type") != "Formula":
            continue
        key = (qid, db, node.get("id"))
        deps = DEPENDS_ON_ADDITIONS.get(key)
        if not deps:
            continue
        present = node.get("depends_on") or []
        merged = list(present)
        for d in deps:
            if d not in merged:
                merged.append(d)
        node["depends_on"] = merged

    return item


def migrate_file(path: Path) -> tuple[int, int]:
    """Migrate to_annotate.json in place. Returns (total_items, items_changed)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items", [])
    changed = 0
    for it in items:
        before = json.dumps(it, sort_keys=True, ensure_ascii=False)
        migrate_item(it)
        after = json.dumps(it, sort_keys=True, ensure_ascii=False)
        if before != after:
            changed += 1
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(items), changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate annotation/to_annotate.json from schema v1 to v2."
    )
    parser.add_argument("path", help="Path to to_annotate.json.")
    args = parser.parse_args()
    total, changed = migrate_file(Path(args.path))
    print(f"Migrated {changed}/{total} items in {args.path}.")


if __name__ == "__main__":
    main()
