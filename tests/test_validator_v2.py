"""Unit tests for the v2 schema additions in validate_annotations.py.

Run with:
  .venv/bin/python -m unittest tests.test_validator_v2 -v
"""
from __future__ import annotations

import os
import sys
import unittest

# Make the project root importable so we can import validate_annotations.py.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from validate_annotations import validate_item  # noqa: E402


def _item(nodes):
    return {
        "question_id": 1,
        "db_id": "x",
        "raw_evidence": "",
        "nodes": nodes,
        "notes": "",
    }


class ColumnAliasNode(unittest.TestCase):
    def test_single_binding_valid(self):
        node = {"id": "a1", "type": "ColumnAlias", "name": "hero name",
                "bindings": [{"table": "superhero", "column": "superhero_name"}]}
        self.assertEqual([], validate_item(_item([node]), 0))

    def test_multi_binding_valid(self):
        node = {"id": "a1", "type": "ColumnAlias", "name": "hometown",
                "bindings": [
                    {"table": "zip_code", "column": "city"},
                    {"table": "zip_code", "column": "county"},
                    {"table": "zip_code", "column": "state"}]}
        self.assertEqual([], validate_item(_item([node]), 0))

    def test_missing_bindings_field_fails(self):
        node = {"id": "a1", "type": "ColumnAlias", "name": "x"}
        errs = validate_item(_item([node]), 0)
        self.assertTrue(any("missing field 'bindings'" in e for e in errs), errs)

    def test_empty_bindings_fails(self):
        node = {"id": "a1", "type": "ColumnAlias", "name": "x", "bindings": []}
        errs = validate_item(_item([node]), 0)
        self.assertTrue(any("non-empty list" in e for e in errs), errs)

    def test_bindings_not_a_list_fails(self):
        node = {"id": "a1", "type": "ColumnAlias", "name": "x",
                "bindings": {"table": "t", "column": "c"}}
        errs = validate_item(_item([node]), 0)
        self.assertTrue(any("non-empty list" in e for e in errs), errs)

    def test_binding_missing_column_fails(self):
        node = {"id": "a1", "type": "ColumnAlias", "name": "x",
                "bindings": [{"table": "t"}]}
        errs = validate_item(_item([node]), 0)
        self.assertTrue(any("bindings[0] missing 'column'" in e for e in errs), errs)

    def test_binding_missing_table_fails(self):
        node = {"id": "a1", "type": "ColumnAlias", "name": "x",
                "bindings": [{"column": "c"}]}
        errs = validate_item(_item([node]), 0)
        self.assertTrue(any("bindings[0] missing 'table'" in e for e in errs), errs)


class GroundingValueField(unittest.TestCase):
    def test_value_field_accepted(self):
        f = {"id": "f1", "type": "Formula", "name": "pct carcinogenic",
             "expression": "SUM(label = '+') / COUNT(*) * 100",
             "grounding": [{"term": "carcinogenic", "table": "molecule",
                            "column": "label", "value": "+"}]}
        self.assertEqual([], validate_item(_item([f]), 0))

    def test_grounding_still_requires_term_table_column(self):
        f = {"id": "f1", "type": "Formula", "name": "x",
             "expression": "x",
             "grounding": [{"value": "+"}]}
        errs = validate_item(_item([f]), 0)
        self.assertTrue(any("grounding[0] missing 'term'" in e for e in errs), errs)
        self.assertTrue(any("grounding[0] missing 'table'" in e for e in errs), errs)
        self.assertTrue(any("grounding[0] missing 'column'" in e for e in errs), errs)


class FormulaDependsOn(unittest.TestCase):
    def test_depends_on_resolved_passes(self):
        r = {"id": "r1", "type": "Rule", "name": "x", "condition": "x",
             "grounding": [{"term": "t", "table": "t", "column": "c"}]}
        f = {"id": "f1", "type": "Formula", "name": "x", "expression": "x",
             "grounding": [{"term": "t", "table": "t", "column": "c"}],
             "depends_on": ["r1"]}
        self.assertEqual([], validate_item(_item([r, f]), 0))

    def test_depends_on_dangling_fails(self):
        f = {"id": "f1", "type": "Formula", "name": "x", "expression": "x",
             "grounding": [{"term": "t", "table": "t", "column": "c"}],
             "depends_on": ["zzz"]}
        errs = validate_item(_item([f]), 0)
        self.assertTrue(any("depends_on[0]='zzz'" in e for e in errs), errs)

    def test_depends_on_not_a_list_fails(self):
        f = {"id": "f1", "type": "Formula", "name": "x", "expression": "x",
             "grounding": [{"term": "t", "table": "t", "column": "c"}],
             "depends_on": "r1"}
        errs = validate_item(_item([f]), 0)
        self.assertTrue(any("depends_on must be a list" in e for e in errs), errs)

    def test_depends_on_absent_is_fine(self):
        f = {"id": "f1", "type": "Formula", "name": "x", "expression": "x",
             "grounding": [{"term": "t", "table": "t", "column": "c"}]}
        self.assertEqual([], validate_item(_item([f]), 0))


class DefinedByTypeWhitelist(unittest.TestCase):
    def test_concept_defined_by_columnalias_passes(self):
        a = {"id": "a1", "type": "ColumnAlias", "name": "x",
             "bindings": [{"table": "t", "column": "c"}]}
        c = {"id": "c1", "type": "Concept", "name": "x", "defined_by": "a1"}
        self.assertEqual([], validate_item(_item([a, c]), 0))

    def test_concept_defined_by_another_concept_fails(self):
        c1 = {"id": "c1", "type": "Concept", "name": "x"}
        c2 = {"id": "c2", "type": "Concept", "name": "y", "defined_by": "c1"}
        errs = validate_item(_item([c1, c2]), 0)
        self.assertTrue(any("defined_by 'c1' points to 'Concept'" in e for e in errs), errs)

    def test_concept_defined_by_dangling_id_fails(self):
        c = {"id": "c1", "type": "Concept", "name": "x", "defined_by": "zzz"}
        errs = validate_item(_item([c]), 0)
        self.assertTrue(any("defined_by 'zzz' references an unknown node id" in e for e in errs), errs)


if __name__ == "__main__":
    unittest.main()
