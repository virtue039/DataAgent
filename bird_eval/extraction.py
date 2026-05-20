"""LLM-driven extractor: BIRD evidence string -> list of v2 typed nodes.

Public surface: extract(evidence, db_id, ddl, llm).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

# Path to the schema doc; the system prompt is generated from it at import-time
# so the prompt automatically reflects any schema change.
_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "annotation" / "node_schema.json"


@lru_cache(maxsize=1)
def _load_schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _build_system_prompt() -> str:
    """Distill the v2 schema doc into a compact system message."""
    schema = _load_schema()
    nt = schema["node_types"]
    lines = [
        "You extract typed business-knowledge nodes from BIRD-style natural-language",
        "evidence strings, grounding each node to a physical SQLite database schema.",
        "",
        "Node types (and required fields):",
        f"- Concept: {nt['Concept']['required']}. Optional: 'definition', 'defined_by' (id of a Formula/Rule/ValueMap/ColumnAlias node in the same record).",
        f"- ColumnAlias: {nt['ColumnAlias']['required']}. 'bindings' is a non-empty list of {{table, column}}.",
        f"- Formula: {nt['Formula']['required']}. Optional 'depends_on' is a list of ids of sibling Rule/Formula/ValueMap/ColumnAlias nodes; joint retrieval must co-fetch these.",
        f"- ValueMap: {nt['ValueMap']['required']}.",
        f"- Rule: {nt['Rule']['required']}. Optional 'name'.",
        "",
        "Grounding rules:",
        "- Every (table, column) in a node's grounding[] or ColumnAlias.bindings[] must exist in the schema DDL.",
        "- A grounding entry without 'value' refers to the column itself (SELECT/GROUP-BY role).",
        "- A grounding entry with 'value' refers to a filter predicate (col = value).",
        "- Both can coexist for the same column; the value-bearing entry is the filter, the bare entry is the projection.",
        "",
        "Edges:",
        "- Concept.defined_by points to one Formula/Rule/ValueMap/ColumnAlias in the same record.",
        "- Formula.depends_on points to a list of Rule/Formula/ValueMap/ColumnAlias in the same record.",
        "",
        "Output:",
        "- A single JSON array of node dicts wrapped in a ```json fence.",
        "- Nothing else outside the fence.",
        "- The array may be empty if the evidence contains no extractable nodes.",
    ]
    return "\n".join(lines)


def _build_user_message(evidence: str, db_id: str, ddl: str) -> str:
    """Compose the per-call user message: examples + task + output reminder."""
    schema = _load_schema()
    parts: list[str] = ["## Examples", ""]
    for ex in schema["examples"]:
        parts.append(f"Evidence: {ex.get('raw_evidence', '')}")
        parts.append("Nodes:")
        parts.append("```json")
        parts.append(json.dumps(ex.get("nodes", []), indent=2, ensure_ascii=False))
        parts.append("```")
        notes = ex.get("notes") or ""
        if notes:
            parts.append(f"Notes: {notes}")
        parts.append("")
    parts += [
        "## Task",
        "",
        f"Database id: `{db_id}`",
        "",
        "Schema (DDL):",
        "```sql",
        ddl,
        "```",
        "",
        f"Evidence: {evidence}",
        "",
        "## Output",
        "",
        "Emit a single JSON array of node dicts inside a ```json fence. Nothing else.",
    ]
    return "\n".join(parts)
