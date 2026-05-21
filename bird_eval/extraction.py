"""LLM-driven extractor: BIRD evidence string -> list of v2 typed nodes.

Public surface: extract(evidence, db_id, ddl, llm).
"""
from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

from .ddl import parse_ddl

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from validate_annotations import validate_item  # noqa: E402

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
        "When to emit a Concept:",
        "- If the evidence names an abstract business term whose meaning is operationalized",
        "  by a Formula, Rule, ValueMap, or ColumnAlias, emit BOTH:",
        "  (a) the operationalizing node (Formula/Rule/ValueMap/ColumnAlias) AND",
        "  (b) a Concept node with `defined_by` pointing at (a).",
        "- Example: evidence 'high-value customer means total_spend > 1000' yields a",
        "  Concept 'high-value customer' with defined_by pointing at a Rule whose",
        "  condition is 'total_spend > 1000'.",
        "- A Concept node alone (without defined_by) is acceptable when the term has",
        "  no clean operationalization. Do NOT emit a Concept that merely wraps a",
        "  ColumnAlias or ValueMap whose name already captures the term -- emit the",
        "  grounding node only. This prohibition does NOT apply to Formula or Rule:",
        "  always emit BOTH a Concept and the Formula/Rule that operationalizes it.",
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


_FENCE_RE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL | re.IGNORECASE)
_BARE_ARRAY_RE = re.compile(r"\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]", re.DOTALL)


def _parse_json_fence(raw: str) -> list[dict]:
    """Pull a JSON array of node dicts out of the LLM response.

    Tries the ```json fence first; falls back to the first balanced `[...]`
    substring. Raises ValueError if neither parses.
    """
    m = _FENCE_RE.search(raw)
    if m:
        return json.loads(m.group(1))
    m = _BARE_ARRAY_RE.search(raw)
    if m:
        return json.loads(m.group(0))
    raise ValueError("No JSON array found in LLM response")


def _sanitize_grounding(nodes: list[dict], ddl: str) -> tuple[list[dict], dict[str, int]]:
    """Drop (table, column) references that don't appear in the DDL.

    Per-node behavior:
    - ValueMap: dropped wholesale if (table, column) is invalid.
    - ColumnAlias: invalid bindings are dropped; node is dropped if no bindings survive.
    - Formula / Rule: invalid grounding entries are dropped; node is dropped if no grounding survives.
    - Concept: passthrough (no grounding).

    Returns (cleaned_nodes, {"dropped_groundings": int, "dropped_nodes": int}).
    """
    tables = parse_ddl(ddl)
    stats = {"dropped_groundings": 0, "dropped_nodes": 0}
    cleaned: list[dict] = []
    for node in nodes:
        ntype = node.get("type")
        if ntype == "ValueMap":
            t, c = node.get("table"), node.get("column")
            if t in tables and c in tables[t]:
                cleaned.append(node)
            else:
                stats["dropped_nodes"] += 1
            continue
        if ntype == "ColumnAlias":
            kept = []
            for b in node.get("bindings", []) or []:
                if b.get("table") in tables and b.get("column") in tables[b["table"]]:
                    kept.append(b)
                else:
                    stats["dropped_groundings"] += 1
            if kept:
                node = {**node, "bindings": kept}
                cleaned.append(node)
            else:
                stats["dropped_nodes"] += 1
            continue
        if ntype in ("Formula", "Rule"):
            kept = []
            for g in node.get("grounding", []) or []:
                t, c = g.get("table"), g.get("column")
                if t in tables and c in tables[t]:
                    kept.append(g)
                else:
                    stats["dropped_groundings"] += 1
            if kept:
                node = {**node, "grounding": kept}
                cleaned.append(node)
            else:
                stats["dropped_nodes"] += 1
            continue
        # Concept and any unknown types: passthrough.
        cleaned.append(node)
    return cleaned, stats


def _try_parse_and_validate(raw: str) -> tuple[list[dict], list[str]]:
    """Parse raw LLM output and structurally validate the node list.

    Returns (nodes, error_list). If error_list is non-empty, the nodes are
    suspect and a repair shot is warranted.
    """
    try:
        nodes = _parse_json_fence(raw)
    except (ValueError, json.JSONDecodeError) as e:
        return [], [f"could not parse JSON array: {e}"]
    if not isinstance(nodes, list):
        return [], ["top-level JSON value must be an array"]
    # Validate via the existing v2 validator. We wrap nodes in a synthetic
    # record so validate_item gets all the fields it expects.
    record = {
        "question_id": -1, "db_id": "", "raw_evidence": "",
        "nodes": nodes, "notes": "",
    }
    errors = validate_item(record, 0)
    return nodes, errors


def _build_repair_message(original_user: str, bad_output: str, errors: list[str]) -> str:
    """User message for the single repair shot."""
    return (
        original_user
        + "\n\n## Repair instructions\n"
        + "Your previous response had the following problems:\n"
        + "\n".join(f"- {e}" for e in errors[:20])
        + "\n\nYour previous response was:\n"
        + "```\n" + bad_output + "\n```\n\n"
        + "Emit a corrected JSON array of nodes inside a ```json fence. "
          "Output only the fenced JSON, nothing else."
    )


def extract(evidence: str, db_id: str, ddl: str, llm) -> tuple[list[dict], dict]:
    """Run the extractor on one evidence string. See spec §3 for semantics.

    Returns (nodes, sanitize_stats):
    - nodes: list of v2 typed-node dicts (possibly empty).
    - sanitize_stats: {"dropped_groundings": int, "dropped_nodes": int} from
      _sanitize_grounding; both zero when no LLM call was made or when
      everything was already valid.

    At most 2 LLM calls are made per invocation (initial + one repair).
    """
    system = _build_system_prompt()
    user = _build_user_message(evidence, db_id, ddl)

    raw = llm.complete(system, user)
    nodes, errors = _try_parse_and_validate(raw)
    if errors:
        repair_user = _build_repair_message(user, raw, errors)
        raw = llm.complete(system, repair_user)
        nodes, errors = _try_parse_and_validate(raw)
        if errors:
            return [], {"dropped_groundings": 0, "dropped_nodes": 0}

    cleaned, stats = _sanitize_grounding(nodes, ddl)
    return cleaned, stats
