"""Unit tests for bird_eval/extraction.py.

Run with:
  .venv/bin/python -m unittest tests.test_extraction -v
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bird_eval.extraction import (  # noqa: E402
    _build_system_prompt,
    _build_user_message,
    _parse_json_fence,
    _sanitize_grounding,
    extract,
)


class SystemPrompt(unittest.TestCase):
    def test_contains_all_five_node_types(self):
        p = _build_system_prompt()
        for t in ("Concept", "Formula", "ValueMap", "Rule", "ColumnAlias"):
            self.assertIn(t, p, f"system prompt missing node type {t!r}")

    def test_contains_both_edge_names(self):
        p = _build_system_prompt()
        self.assertIn("defined_by", p)
        self.assertIn("depends_on", p)

    def test_mentions_grounding_value_field(self):
        p = _build_system_prompt()
        # Grounding rules must reference the optional 'value' field
        self.assertIn("value", p)
        self.assertIn("grounding", p)

    def test_includes_when_to_emit_concept_section(self):
        p = _build_system_prompt()
        # F2 fix: the prompt must explicitly tell the LLM to emit a Concept
        # node alongside the Formula/Rule that operationalizes it.
        self.assertIn("When to emit a Concept", p)
        self.assertIn("operationalized", p.lower())
        self.assertIn("defined_by", p)


class UserMessage(unittest.TestCase):
    def test_includes_all_five_schema_examples(self):
        msg = _build_user_message("dummy evidence", "db_x", "CREATE TABLE t (c INTEGER);")
        # Each schema-doc example has a distinctive raw_evidence snippet
        for snippet in (
            "Eligible free rate for K-12",
            "POPLATEK MESICNE",
            "abnormal white blood cell count",
            "magic card market name",
            "hometown refers to city, county, state",
        ):
            self.assertIn(snippet, msg, f"user message missing example snippet {snippet!r}")

    def test_includes_task_inputs_verbatim(self):
        msg = _build_user_message(
            "my evidence here", "my_db_id", "CREATE TABLE my_table (my_col INTEGER);"
        )
        self.assertIn("my evidence here", msg)
        self.assertIn("my_db_id", msg)
        self.assertIn("CREATE TABLE my_table", msg)


class JsonFenceParser(unittest.TestCase):
    def test_parses_explicit_json_fence(self):
        raw = "Some prose ```json\n[{\"id\": \"a1\", \"type\": \"Concept\", \"name\": \"x\"}]\n```\nTail"
        self.assertEqual(
            [{"id": "a1", "type": "Concept", "name": "x"}],
            _parse_json_fence(raw),
        )

    def test_parses_unfenced_array_as_fallback(self):
        raw = "Prose [{\"id\": \"a1\", \"type\": \"Concept\", \"name\": \"x\"}] tail"
        self.assertEqual(
            [{"id": "a1", "type": "Concept", "name": "x"}],
            _parse_json_fence(raw),
        )

    def test_no_array_raises_value_error(self):
        with self.assertRaises(ValueError):
            _parse_json_fence("just some prose with no JSON")


# Minimal DDL fixture for sanitizer tests.
_DDL = """
CREATE TABLE good_table (
    real_col INTEGER,
    another_real_col TEXT
);
CREATE TABLE other (
    x INTEGER
);
"""


class GroundingSanitizer(unittest.TestCase):
    def test_valid_node_kept_untouched(self):
        nodes = [
            {"id": "f1", "type": "Formula", "name": "x", "expression": "x",
             "grounding": [{"term": "real_col", "table": "good_table", "column": "real_col"}]}
        ]
        out, stats = _sanitize_grounding(nodes, _DDL)
        self.assertEqual(nodes, out)
        self.assertEqual(stats, {"dropped_groundings": 0, "dropped_nodes": 0})

    def test_drops_hallucinated_grounding_entry_only(self):
        nodes = [
            {"id": "f1", "type": "Formula", "name": "x", "expression": "x",
             "grounding": [
                 {"term": "real_col", "table": "good_table", "column": "real_col"},
                 {"term": "fake", "table": "good_table", "column": "no_such_col"}]}
        ]
        out, stats = _sanitize_grounding(nodes, _DDL)
        self.assertEqual(1, len(out))
        self.assertEqual(1, len(out[0]["grounding"]))
        self.assertEqual("real_col", out[0]["grounding"][0]["column"])
        self.assertEqual(1, stats["dropped_groundings"])
        self.assertEqual(0, stats["dropped_nodes"])

    def test_drops_formula_when_all_grounding_invalid(self):
        nodes = [
            {"id": "f1", "type": "Formula", "name": "x", "expression": "x",
             "grounding": [
                 {"term": "fake1", "table": "good_table", "column": "no_such_col"},
                 {"term": "fake2", "table": "missing_table", "column": "x"}]}
        ]
        out, stats = _sanitize_grounding(nodes, _DDL)
        self.assertEqual([], out)
        self.assertEqual(1, stats["dropped_nodes"])

    def test_columnalias_drops_invalid_bindings(self):
        nodes = [
            {"id": "a1", "type": "ColumnAlias", "name": "x",
             "bindings": [
                 {"table": "good_table", "column": "real_col"},
                 {"table": "missing_table", "column": "x"}]}
        ]
        out, stats = _sanitize_grounding(nodes, _DDL)
        self.assertEqual(1, len(out))
        self.assertEqual(
            [{"table": "good_table", "column": "real_col"}], out[0]["bindings"]
        )
        self.assertEqual(1, stats["dropped_groundings"])

    def test_valuemap_drops_when_table_or_column_invalid(self):
        nodes = [
            {"id": "v1", "type": "ValueMap", "name": "x",
             "table": "good_table", "column": "no_such_col", "value": "v"}
        ]
        out, stats = _sanitize_grounding(nodes, _DDL)
        self.assertEqual([], out)
        self.assertEqual(1, stats["dropped_nodes"])

    def test_gold_standard_roundtrip_keeps_all_nodes(self):
        """Sanity check: the 40-item gold standard is by construction grounded.
        Running it through the sanitizer (against the embedded DDL) must drop
        zero entries -- otherwise the sanitizer is over-zealous."""
        path = os.path.join(ROOT, "annotation", "to_annotate.json")
        import json as _json
        with open(path) as f:
            data = _json.load(f)
        total_dropped_g = 0
        total_dropped_n = 0
        for it in data["items"]:
            ddl = data["schemas"][it["db_id"]]
            _, stats = _sanitize_grounding(it["nodes"], ddl)
            total_dropped_g += stats["dropped_groundings"]
            total_dropped_n += stats["dropped_nodes"]
        self.assertEqual(0, total_dropped_g, "sanitizer dropped a valid grounding")
        self.assertEqual(0, total_dropped_n, "sanitizer dropped a valid node")


class _MockLLMClient:
    """In-memory stand-in for bird_eval.llm.LLMClient.

    Returns canned responses in order. Tests use this to drive the repair
    branch deterministically without hitting a real endpoint.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def complete(self, system, user, retries=3):
        self.calls.append({"system": system, "user": user})
        if not self._responses:
            raise AssertionError("MockLLMClient exhausted; no canned response left")
        return self._responses.pop(0)


_VALID_RESPONSE = """```json
[
  {"id": "a1", "type": "ColumnAlias", "name": "x",
   "bindings": [{"table": "good_table", "column": "real_col"}]}
]
```"""

_MALFORMED_JSON_RESPONSE = "Here is the answer: not_valid_json_at_all"

_VALIDATION_FAIL_RESPONSE = """```json
[
  {"id": "a1", "type": "ColumnAlias", "name": "x"}
]
```"""  # missing required 'bindings' field


class ExtractOrchestrator(unittest.TestCase):
    def test_happy_path_returns_nodes_single_call(self):
        llm = _MockLLMClient([_VALID_RESPONSE])
        nodes = extract("ev", "db_x", _DDL, llm)
        self.assertEqual(1, len(llm.calls), "should not have invoked repair")
        self.assertEqual(1, len(nodes))
        self.assertEqual("ColumnAlias", nodes[0]["type"])

    def test_repair_invoked_on_parse_failure(self):
        llm = _MockLLMClient([_MALFORMED_JSON_RESPONSE, _VALID_RESPONSE])
        nodes = extract("ev", "db_x", _DDL, llm)
        self.assertEqual(2, len(llm.calls), "expected exactly one repair shot")
        self.assertEqual(1, len(nodes))

    def test_repair_invoked_on_validation_failure(self):
        llm = _MockLLMClient([_VALIDATION_FAIL_RESPONSE, _VALID_RESPONSE])
        nodes = extract("ev", "db_x", _DDL, llm)
        self.assertEqual(2, len(llm.calls))
        self.assertEqual(1, len(nodes))

    def test_both_calls_fail_returns_empty_list(self):
        llm = _MockLLMClient([_MALFORMED_JSON_RESPONSE, _MALFORMED_JSON_RESPONSE])
        nodes = extract("ev", "db_x", _DDL, llm)
        self.assertEqual(2, len(llm.calls), "must cap at 2 LLM calls")
        self.assertEqual([], nodes)


if __name__ == "__main__":
    unittest.main()
