"""Validate a typed-node annotation file against annotation/node_schema.json.

Checks structure only (node types, required fields, unique ids, grounding
shape, defined_by references) -- not annotation quality. Empty 'nodes' lists
are allowed, so this also passes on a fresh, unannotated template.

Usage:
  python validate_annotations.py annotation/to_annotate.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RECORD_REQUIRED = ["question_id", "db_id", "raw_evidence", "nodes", "notes"]

NODE_REQUIRED = {
    "Concept": ["id", "type", "name"],
    "Formula": ["id", "type", "name", "expression", "grounding"],
    "ValueMap": ["id", "type", "name", "table", "column", "value"],
    "Rule": ["id", "type", "condition", "grounding"],
}


def validate_item(item: dict, idx: int) -> list[str]:
    errors: list[str] = []
    tag = f"item[{idx}] (qid={item.get('question_id', '?')})"

    for field in RECORD_REQUIRED:
        if field not in item:
            errors.append(f"{tag}: missing record field '{field}'")

    nodes = item.get("nodes", [])
    if not isinstance(nodes, list):
        errors.append(f"{tag}: 'nodes' must be a list")
        return errors

    ids: list[str] = []
    for j, node in enumerate(nodes):
        ntag = f"{tag} node[{j}]"
        ntype = node.get("type")
        if ntype not in NODE_REQUIRED:
            errors.append(
                f"{ntag}: invalid type {ntype!r} (expected one of {list(NODE_REQUIRED)})"
            )
            continue
        for field in NODE_REQUIRED[ntype]:
            if field not in node:
                errors.append(f"{ntag}: {ntype} missing field '{field}'")
        if node.get("id") is not None:
            ids.append(node["id"])
        for k, ground in enumerate(node.get("grounding", []) or []):
            for gfield in ("term", "table", "column"):
                if gfield not in ground:
                    errors.append(f"{ntag}: grounding[{k}] missing '{gfield}'")

    for dup in {i for i in ids if ids.count(i) > 1}:
        errors.append(f"{tag}: duplicate node id {dup!r}")

    idset = set(ids)
    for j, node in enumerate(nodes):
        ref = node.get("defined_by")
        if ref is not None and ref not in idset:
            errors.append(f"{tag} node[{j}]: defined_by {ref!r} references an unknown node id")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a typed-node annotation file.")
    parser.add_argument("path", help="Annotation JSON (sample_evidence.py output, nodes filled).")
    args = parser.parse_args()

    data = json.loads(Path(args.path).read_text(encoding="utf-8"))
    items = data.get("items", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise SystemExit("Could not find an 'items' list in the annotation file.")

    all_errors: list[str] = []
    n_annotated = 0
    for idx, item in enumerate(items):
        if item.get("nodes"):
            n_annotated += 1
        all_errors.extend(validate_item(item, idx))

    print(f"Checked {len(items)} items ({n_annotated} with >=1 node).")
    if all_errors:
        print(f"\n{len(all_errors)} problem(s):")
        for err in all_errors:
            print(f"  - {err}")
        sys.exit(1)
    print("OK: all items conform to annotation/node_schema.json.")


if __name__ == "__main__":
    main()
