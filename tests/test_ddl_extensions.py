"""Unit tests for bird_eval/ddl.py extensions (parse_ddl_typed, parse_fks).

Run with:
  .venv/bin/python -m unittest tests.test_ddl_extensions -v
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bird_eval.ddl import parse_ddl_typed, parse_fks  # noqa: E402


_DDL_INLINE_FK = """
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT,
    price REAL
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    product_id INTEGER REFERENCES products (product_id),
    quantity INTEGER
);
"""

_DDL_CLAUSAL_FK = """
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT
);
CREATE TABLE shipments (
    shipment_id INTEGER PRIMARY KEY,
    product_id INTEGER,
    FOREIGN KEY (product_id) REFERENCES products (product_id)
);
"""

_DDL_NO_FK = """
CREATE TABLE standalone (
    id INTEGER,
    label TEXT
);
"""

_DDL_QUOTED = """
CREATE TABLE "frpm" (
    `Academic Year` TEXT,
    `Free Meal Count (K-12)` REAL,
    `Enrollment (K-12)` REAL
);
"""


class ParseDdlTyped(unittest.TestCase):
    def test_returns_mapping_with_types(self):
        out = parse_ddl_typed(_DDL_INLINE_FK)
        self.assertIn(("products", "product_id"), out)
        self.assertEqual("INTEGER", out[("products", "product_id")]["type"])
        self.assertEqual("REAL",    out[("products", "price")]["type"])
        self.assertEqual("TEXT",    out[("products", "name")]["type"])

    def test_preserves_columns_with_spaces_and_punctuation(self):
        out = parse_ddl_typed(_DDL_QUOTED)
        self.assertIn(("frpm", "Academic Year"), out)
        self.assertIn(("frpm", "Free Meal Count (K-12)"), out)
        self.assertEqual("REAL", out[("frpm", "Enrollment (K-12)")]["type"])


class ParseFks(unittest.TestCase):
    def test_inline_references_picked_up(self):
        out = parse_fks(_DDL_INLINE_FK)
        self.assertIn((("orders", "product_id"), ("products", "product_id")), out)

    def test_clausal_foreign_key_picked_up(self):
        out = parse_fks(_DDL_CLAUSAL_FK)
        self.assertIn((("shipments", "product_id"), ("products", "product_id")), out)

    def test_no_fks_returns_empty(self):
        self.assertEqual((), parse_fks(_DDL_NO_FK))


if __name__ == "__main__":
    unittest.main()
