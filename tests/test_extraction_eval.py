"""Unit tests for bird_eval/extraction_eval.py.

Run with:
  .venv/bin/python -m unittest tests.test_extraction_eval -v
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bird_eval.extraction_eval import match  # noqa: E402


class NodeMatch(unittest.TestCase):
    def test_different_types_never_match(self):
        a = {"type": "Concept", "name": "x"}
        b = {"type": "Formula", "name": "x", "expression": "x"}
        self.assertFalse(match(a, b))

    def test_concept_matches_by_name_casefolded(self):
        a = {"type": "Concept", "name": "High Value Customer"}
        b = {"type": "Concept", "name": "  HIGH value customer  "}
        self.assertTrue(match(a, b))

    def test_concept_no_match_on_different_name(self):
        a = {"type": "Concept", "name": "high value customer"}
        b = {"type": "Concept", "name": "low value customer"}
        self.assertFalse(match(a, b))

    def test_columnalias_matches_by_binding_set(self):
        a = {"type": "ColumnAlias", "name": "x",
             "bindings": [{"table": "t1", "column": "c1"},
                          {"table": "t2", "column": "c2"}]}
        b = {"type": "ColumnAlias", "name": "different name",
             "bindings": [{"table": "T2", "column": "C2"},
                          {"table": "T1", "column": "C1"}]}
        self.assertTrue(match(a, b), "binding sets equal modulo order and case")

    def test_columnalias_no_match_when_binding_sets_differ(self):
        a = {"type": "ColumnAlias", "name": "x",
             "bindings": [{"table": "t1", "column": "c1"}]}
        b = {"type": "ColumnAlias", "name": "x",
             "bindings": [{"table": "t1", "column": "c2"}]}
        self.assertFalse(match(a, b))

    def test_valuemap_matches_by_table_column_value_tuple(self):
        a = {"type": "ValueMap", "name": "x", "table": "T", "column": "C", "value": "V"}
        b = {"type": "ValueMap", "name": "y", "table": "t", "column": "c", "value": "v"}
        self.assertTrue(match(a, b))

    def test_formula_matches_by_name_only(self):
        # Two formulas with same name but different expressions and grounding
        # still match (grounding fidelity is scored separately).
        a = {"type": "Formula", "name": "X rate", "expression": "a/b",
             "grounding": [{"term": "a", "table": "t", "column": "ca"}]}
        b = {"type": "Formula", "name": "x rate", "expression": "c/d",
             "grounding": [{"term": "c", "table": "u", "column": "cc"}]}
        self.assertTrue(match(a, b))

    def test_rule_falls_back_to_condition_when_no_name(self):
        a = {"type": "Rule", "condition": "TP < 6.0"}
        b = {"type": "Rule", "condition": "tp < 6.0  "}
        self.assertTrue(match(a, b))


if __name__ == "__main__":
    unittest.main()
