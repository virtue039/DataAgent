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

from bird_eval.extraction_eval import match, match_pairs, compare_items, aggregate  # noqa: E402


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

    def test_formula_matches_by_name_when_no_expression(self):
        # Two formulas with same name but no expressions still match
        # (name is used as fallback when expression is missing).
        a = {"type": "Formula", "name": "X rate",
             "grounding": [{"term": "a", "table": "t", "column": "ca"}]}
        b = {"type": "Formula", "name": "x rate",
             "grounding": [{"term": "c", "table": "u", "column": "cc"}]}
        self.assertTrue(match(a, b))

    def test_rule_falls_back_to_condition_when_no_name(self):
        a = {"type": "Rule", "condition": "TP < 6.0"}
        b = {"type": "Rule", "condition": "tp < 6.0  "}
        self.assertTrue(match(a, b))

    def test_rule_matches_by_condition_when_name_paraphrases(self):
        # Both Rules have different paraphrased names but identical conditions;
        # the deterministic condition wins.
        a = {"type": "Rule", "name": "EDHRec rank below 100",
             "condition": "edhrecRank < 100"}
        b = {"type": "Rule", "name": "below 100 on EDHRec",
             "condition": "edhrecRank < 100"}
        self.assertTrue(match(a, b))

    def test_rule_no_match_when_condition_differs(self):
        # Same name, different conditions => no match.
        a = {"type": "Rule", "name": "low rank", "condition": "rank < 100"}
        b = {"type": "Rule", "name": "low rank", "condition": "rank < 200"}
        self.assertFalse(match(a, b))

    def test_formula_matches_by_expression_when_name_paraphrases(self):
        # Both Formulas have different paraphrased names but identical expressions;
        # the deterministic expression wins (mirrors the Rule fix in commit 79c7e3d).
        a = {"type": "Formula", "name": "Eligible free rate for K-12",
             "expression": "`Free Meal Count (K-12)` / `Enrollment (K-12)`",
             "grounding": [{"term": "Free Meal Count (K-12)",
                            "table": "frpm", "column": "Free Meal Count (K-12)"}]}
        b = {"type": "Formula", "name": "K-12 free meal eligibility ratio",
             "expression": "`Free Meal Count (K-12)` / `Enrollment (K-12)`",
             "grounding": [{"term": "Free Meal Count (K-12)",
                            "table": "frpm", "column": "Free Meal Count (K-12)"}]}
        self.assertTrue(match(a, b))

    def test_formula_no_match_when_expression_differs(self):
        a = {"type": "Formula", "name": "same name", "expression": "a / b",
             "grounding": [{"term": "a", "table": "t", "column": "a"}]}
        b = {"type": "Formula", "name": "same name", "expression": "c / d",
             "grounding": [{"term": "c", "table": "t", "column": "c"}]}
        self.assertFalse(match(a, b))


class MatchPairs(unittest.TestCase):
    def test_one_to_one_assignment(self):
        # Two gold Concepts named A and B; predicted has two A's and one B.
        # Expected: gold[0] (A) matches predicted[0]; gold[1] (B) matches predicted[2].
        gold = [
            {"type": "Concept", "name": "A"},
            {"type": "Concept", "name": "B"},
        ]
        predicted = [
            {"type": "Concept", "name": "a"},
            {"type": "Concept", "name": "a"},
            {"type": "Concept", "name": "B"},
        ]
        pairs = match_pairs(predicted, gold)
        self.assertIn((0, 0), pairs)
        self.assertIn((2, 1), pairs)
        # No gold node is double-matched
        gold_indices = [g for _, g in pairs]
        self.assertEqual(len(gold_indices), len(set(gold_indices)))


class CompareItem(unittest.TestCase):
    def test_perfect_match_zero_extras(self):
        gold = [
            {"type": "Concept", "name": "X"},
            {"type": "ValueMap", "name": "y", "table": "t", "column": "c", "value": "v"},
        ]
        predicted = [dict(g) for g in gold]
        out = compare_items(predicted, gold)
        self.assertEqual(2, out["matched"])
        self.assertEqual(0, out["fp"])
        self.assertEqual(0, out["fn"])
        # tp counted per type
        self.assertEqual(1, out["per_type"]["Concept"]["tp"])
        self.assertEqual(1, out["per_type"]["ValueMap"]["tp"])

    def test_partial_match_counts_fp_and_fn(self):
        gold = [{"type": "Concept", "name": "A"}, {"type": "Concept", "name": "B"}]
        predicted = [{"type": "Concept", "name": "A"}, {"type": "Concept", "name": "C"}]
        out = compare_items(predicted, gold)
        self.assertEqual(1, out["matched"])
        self.assertEqual(1, out["fp"], "C is a false positive")
        self.assertEqual(1, out["fn"], "B is a false negative")


class Aggregate(unittest.TestCase):
    def test_aggregate_computes_p_r_f1(self):
        # Two items: item-0 has 2 matched, 0 fp, 0 fn; item-1 has 1 matched, 1 fp, 1 fn.
        per_item = [
            {"matched": 2, "fp": 0, "fn": 0,
             "per_type": {"Concept": {"tp": 2, "fp": 0, "fn": 0}}, "grounding_jaccards": []},
            {"matched": 1, "fp": 1, "fn": 1,
             "per_type": {"Concept": {"tp": 1, "fp": 1, "fn": 1}}, "grounding_jaccards": []},
        ]
        s = aggregate(per_item)
        # Overall: tp=3, fp=1, fn=1 -> P = 3/4 = 0.75, R = 3/4 = 0.75, F1 = 0.75
        self.assertAlmostEqual(0.75, s["overall"]["precision"], places=4)
        self.assertAlmostEqual(0.75, s["overall"]["recall"], places=4)
        self.assertAlmostEqual(0.75, s["overall"]["f1"], places=4)
        # Per-type same numbers (only Concept appears).
        self.assertAlmostEqual(0.75, s["by_type"]["Concept"]["f1"], places=4)

    def test_aggregate_empty_predicted_recall_zero(self):
        per_item = [{
            "matched": 0, "fp": 0, "fn": 3,
            "per_type": {"Concept": {"tp": 0, "fp": 0, "fn": 3}},
            "grounding_jaccards": [],
        }]
        s = aggregate(per_item)
        self.assertEqual(0.0, s["overall"]["recall"])
        # Precision is 0/0 -> we choose to return 0.0 in that case.
        self.assertEqual(0.0, s["overall"]["precision"])
        self.assertEqual(0.0, s["overall"]["f1"])


if __name__ == "__main__":
    unittest.main()
