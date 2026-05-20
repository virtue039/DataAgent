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


if __name__ == "__main__":
    unittest.main()
