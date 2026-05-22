"""Unit tests for bird_eval/audit_analysis.py (P4)."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bird_eval.audit_analysis import (  # noqa: E402
    category_histogram,
    per_db_breakdown,
    per_difficulty_breakdown,
    pick_case_studies,
)


def _judgment(qid: int, db_id: str, difficulty: str, category: str) -> dict:
    return {"question_id": qid, "db_id": db_id, "difficulty": difficulty,
            "category": category, "reason": "test", "evidence_quote": "..."}


class CategoryHistogram(unittest.TestCase):
    def test_aggregates_counts_with_wilson_ci(self):
        judgments = [
            _judgment(1, "a", "simple", "specificity_loss"),
            _judgment(2, "a", "simple", "specificity_loss"),
            _judgment(3, "a", "simple", "retrieval_irrelevance"),
            _judgment(4, "b", "moderate", "format_dilution"),
        ]
        rows = category_histogram(judgments)
        # Find by category.
        h = {r["category"]: r for r in rows}
        self.assertEqual(h["specificity_loss"]["count"], 2)
        self.assertEqual(h["retrieval_irrelevance"]["count"], 1)
        self.assertEqual(h["format_dilution"]["count"], 1)
        self.assertAlmostEqual(h["specificity_loss"]["proportion"], 0.5)
        # CI is present and brackets the proportion.
        self.assertIn("ci_lo", h["specificity_loss"])
        self.assertIn("ci_hi", h["specificity_loss"])

    def test_zero_count_categories_still_appear(self):
        # If no qid gets "other", the row still shows up with count=0.
        judgments = [
            _judgment(1, "a", "simple", "specificity_loss"),
        ]
        rows = category_histogram(judgments)
        h = {r["category"]: r for r in rows}
        self.assertEqual(h["other"]["count"], 0)
        self.assertEqual(h["other"]["proportion"], 0.0)


class PerDbAndDifficulty(unittest.TestCase):
    def test_per_db_groups_correctly(self):
        judgments = [
            _judgment(1, "alpha", "simple", "specificity_loss"),
            _judgment(2, "alpha", "moderate", "retrieval_irrelevance"),
            _judgment(3, "beta", "simple", "specificity_loss"),
        ]
        rows = per_db_breakdown(judgments)
        d = {r["db_id"]: r for r in rows}
        self.assertEqual(d["alpha"]["n"], 2)
        self.assertEqual(d["alpha"]["specificity_loss"], 1)
        self.assertEqual(d["beta"]["n"], 1)
        self.assertEqual(d["beta"]["retrieval_irrelevance"], 0)

    def test_pick_case_studies_two_per_category(self):
        # 4 specificity_loss judgments, 2 retrieval_irrelevance, 0 others.
        judgments = [_judgment(i, "x", "simple", "specificity_loss") for i in range(4)]
        judgments += [_judgment(i + 10, "x", "simple", "retrieval_irrelevance")
                      for i in range(2)]
        cases = pick_case_studies(judgments, k_per_category=2)
        # Two per non-empty category, none for empty.
        self.assertEqual(len([c for c in cases if c["category"] == "specificity_loss"]), 2)
        self.assertEqual(len([c for c in cases if c["category"] == "retrieval_irrelevance"]), 2)
        self.assertEqual(len([c for c in cases if c["category"] == "format_dilution"]), 0)
