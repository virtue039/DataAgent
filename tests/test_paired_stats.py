"""Unit tests for bird_eval/paired_stats.py (P4)."""
from __future__ import annotations

import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bird_eval.paired_stats import (  # noqa: E402
    mcnemar_with_correction,
    newcombe_mover_diff_ci,
)


class McNemar(unittest.TestCase):
    def test_textbook_agresti_2002(self):
        # Agresti (2002) Categorical Data Analysis, Section 10.1 example.
        # b = 6 (A correct ∧ B wrong), c = 18 (A wrong ∧ B correct).
        # χ² (with continuity correction) = (|6-18|-1)² / (6+18) = 121/24 ≈ 5.04.
        # p ≈ 0.025 (one-tailed chi² with 1 df).
        chi2, p = mcnemar_with_correction(b=6, c=18)
        self.assertAlmostEqual(chi2, 5.041666, places=4)
        self.assertLess(p, 0.05)
        self.assertGreater(p, 0.01)

    def test_no_discordant_pairs_returns_p_eq_1(self):
        # b == 0 and c == 0: the two settings agree perfectly.
        # χ² is undefined; we return χ²=0, p=1.0 (no evidence of difference).
        chi2, p = mcnemar_with_correction(b=0, c=0)
        self.assertEqual(chi2, 0.0)
        self.assertEqual(p, 1.0)

    def test_large_imbalance_gives_tiny_p(self):
        # 200 vs 5 — overwhelming evidence A != B.
        chi2, p = mcnemar_with_correction(b=200, c=5)
        self.assertGreater(chi2, 100.0)
        self.assertLess(p, 1e-20)


class NewcombeMOVER(unittest.TestCase):
    def test_paired_no_difference(self):
        # 100/200 vs 100/200 = 0% diff. CI should bracket 0 symmetrically.
        # Discordant: b=20, c=20.
        lo, hi = newcombe_mover_diff_ci(
            a_correct=100, a_total=200,
            b_correct=100, b_total=200,
            b_count=20, c_count=20,
        )
        self.assertAlmostEqual((lo + hi) / 2, 0.0, places=2)
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_paired_strong_difference(self):
        # 150/200 vs 50/200 = 50pp diff. CI should NOT bracket 0.
        # Concordant: 50 both correct, 50 both wrong. Discordant: b=100 (A correct ∧ B wrong),
        # c=0 (A wrong ∧ B correct). Sanity: 100/200 - 0/200 = 100pp accuracy on the
        # discordant pairs, scaled by n=200 gives diff = 50pp.
        lo, hi = newcombe_mover_diff_ci(
            a_correct=150, a_total=200,
            b_correct=50, b_total=200,
            b_count=100, c_count=0,
        )
        self.assertGreater(lo, 0.30)  # well above 0
        self.assertLess(hi, 0.70)
