"""Unit tests for bird_eval/joint_graph.py.

Run with:
  .venv/bin/python -m unittest tests.test_joint_graph -v
"""
from __future__ import annotations

import dataclasses
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bird_eval.joint_graph import (  # noqa: E402
    JointGraph,
    _dedup_l1,
    _rewrite_refs,
    _drop_hallucinated,
    _build_indexes,
    build_graph,
    dump_graph,
    load_graph,
)


def _empty_graph() -> JointGraph:
    return JointGraph(
        db_id="example",
        nodes={},
        columns={},
        fk_edges=(),
        by_type={},
        by_column={},
        by_name_token={},
        provenance={},
    )


class Dataclass(unittest.TestCase):
    def test_is_frozen(self):
        g = _empty_graph()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            g.db_id = "other"  # type: ignore[misc]

    def test_equality_is_structural(self):
        a = _empty_graph()
        b = _empty_graph()
        self.assertEqual(a, b)

    def test_carries_all_required_fields(self):
        g = _empty_graph()
        for field in (
            "db_id", "nodes", "columns", "fk_edges",
            "by_type", "by_column", "by_name_token", "provenance",
        ):
            self.assertTrue(hasattr(g, field), f"missing field {field!r}")


# Shared fixture items used in dedup + sanitize + build tests below.

_ITEM_RULE_A = {
    "question_id": 1, "db_id": "shop", "difficulty": "simple",
    "question": "low-priced products",
    "raw_evidence": "low-priced products refers to price < 10",
    "nodes": [
        {"id": "r1", "type": "Rule", "name": "low-priced",
         "condition": "price < 10",
         "grounding": [{"term": "price", "table": "products", "column": "price"}]},
    ],
    "notes": "",
}

_ITEM_RULE_B_SAME_CONDITION = {
    # Same condition as ITEM_RULE_A; different name; same db.
    # Post the 2026-05-20 Rule fix, match() uses condition-primary => dedup.
    "question_id": 7, "db_id": "shop", "difficulty": "simple",
    "question": "cheap items",
    "raw_evidence": "cheap items: price < 10",
    "nodes": [
        {"id": "r1", "type": "Rule", "name": "cheap items",
         "condition": "price < 10",
         "grounding": [
             {"term": "price", "table": "products", "column": "price"},
             # extra (term, col, val) tuple that B has but A doesn't
             {"term": "price", "table": "products", "column": "price", "value": "10"},
         ]},
    ],
    "notes": "",
}

_ITEM_RULE_DISTINCT = {
    # Different condition => no dedup with A or B.
    "question_id": 9, "db_id": "shop", "difficulty": "simple",
    "question": "expensive items",
    "raw_evidence": "expensive items: price >= 100",
    "nodes": [
        {"id": "r1", "type": "Rule", "name": "expensive",
         "condition": "price >= 100",
         "grounding": [{"term": "price", "table": "products", "column": "price"}]},
    ],
    "notes": "",
}


class L1Dedup(unittest.TestCase):
    def test_collapses_same_condition_rules(self):
        canonical_nodes, provenance, _ = _dedup_l1(
            [_ITEM_RULE_A, _ITEM_RULE_B_SAME_CONDITION]
        )
        # exactly one Rule node survives
        rules = [n for n in canonical_nodes.values() if n["type"] == "Rule"]
        self.assertEqual(1, len(rules))
        self.assertTrue(rules[0]["id"].startswith("r"))
        self.assertIn(rules[0]["id"], canonical_nodes)

    def test_provenance_records_both_origins(self):
        canonical_nodes, provenance, _ = _dedup_l1(
            [_ITEM_RULE_A, _ITEM_RULE_B_SAME_CONDITION]
        )
        rule_id = next(k for k, n in canonical_nodes.items() if n["type"] == "Rule")
        prov = provenance[rule_id]
        self.assertEqual({(1, "r1"), (7, "r1")}, set(prov))

    def test_grounding_union_across_records(self):
        canonical_nodes, _, _ = _dedup_l1(
            [_ITEM_RULE_A, _ITEM_RULE_B_SAME_CONDITION]
        )
        rule = next(n for n in canonical_nodes.values() if n["type"] == "Rule")
        # union by (term, table, column, value)
        keys = {(g.get("term"), g.get("table"), g.get("column"), g.get("value"))
                for g in rule["grounding"]}
        self.assertEqual(
            {("price", "products", "price", None),
             ("price", "products", "price", "10")},
            keys,
        )

    def test_distinct_rules_kept_separate(self):
        canonical_nodes, _, _ = _dedup_l1(
            [_ITEM_RULE_A, _ITEM_RULE_DISTINCT]
        )
        rules = [n for n in canonical_nodes.values() if n["type"] == "Rule"]
        self.assertEqual(2, len(rules))
        # canonical ids are deterministic: r1, r2
        rule_ids = sorted(n["id"] for n in rules)
        self.assertEqual(["r1", "r2"], rule_ids)


_TEST_DDL = """
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


class RewriteRefs(unittest.TestCase):
    def test_concept_defined_by_rewritten_to_canonical_id(self):
        items = [{
            "question_id": 1, "db_id": "shop", "difficulty": "simple",
            "question": "", "raw_evidence": "", "notes": "",
            "nodes": [
                {"id": "c1", "type": "Concept", "name": "cheap product",
                 "defined_by": "r1"},
                {"id": "r1", "type": "Rule", "name": "cheap",
                 "condition": "price < 10",
                 "grounding": [{"term": "price", "table": "products", "column": "price"}]},
            ],
        }]
        canon, prov, remap = _dedup_l1(items)
        rewritten = _rewrite_refs(canon, prov, remap)
        c = next(n for n in rewritten.values() if n["type"] == "Concept")
        self.assertIn(c["defined_by"], rewritten)  # resolved to a present canonical id
        self.assertEqual("Rule", rewritten[c["defined_by"]]["type"])

    def test_formula_depends_on_rewritten(self):
        items = [{
            "question_id": 5, "db_id": "shop", "difficulty": "simple",
            "question": "", "raw_evidence": "", "notes": "",
            "nodes": [
                {"id": "r1", "type": "Rule", "name": "y", "condition": "year > 2020",
                 "grounding": [{"term": "year", "table": "orders", "column": "quantity"}]},
                {"id": "f1", "type": "Formula", "name": "x", "expression": "x",
                 "grounding": [{"term": "x", "table": "products", "column": "price"}],
                 "depends_on": ["r1"]},
            ],
        }]
        canon, prov, remap = _dedup_l1(items)
        rewritten = _rewrite_refs(canon, prov, remap)
        f = next(n for n in rewritten.values() if n["type"] == "Formula")
        self.assertEqual(1, len(f["depends_on"]))
        target = rewritten[f["depends_on"][0]]
        self.assertEqual("Rule", target["type"])


class SanityDrop(unittest.TestCase):
    def test_node_with_hallucinated_column_loses_that_grounding(self):
        items = [{
            "question_id": 1, "db_id": "shop", "difficulty": "simple",
            "question": "", "raw_evidence": "", "notes": "",
            "nodes": [
                {"id": "f1", "type": "Formula", "name": "x", "expression": "x",
                 "grounding": [
                     {"term": "price", "table": "products", "column": "price"},
                     {"term": "bogus", "table": "products", "column": "no_such_col"},
                 ]},
            ],
        }]
        canon, _, _ = _dedup_l1(items)
        cleaned = _drop_hallucinated(canon, _TEST_DDL)
        f = next(n for n in cleaned.values() if n["type"] == "Formula")
        self.assertEqual(1, len(f["grounding"]))
        self.assertEqual("price", f["grounding"][0]["column"])


class Indexes(unittest.TestCase):
    def _sample_nodes(self):
        return {
            "f1": {"id": "f1", "type": "Formula",
                   "name": "average product price",
                   "expression": "AVG(price)",
                   "grounding": [{"term": "price", "table": "products", "column": "price"}]},
            "v1": {"id": "v1", "type": "ValueMap",
                   "name": "available",
                   "table": "products", "column": "name", "value": "in_stock"},
            "a1": {"id": "a1", "type": "ColumnAlias",
                   "name": "the product table",
                   "bindings": [{"table": "products", "column": "product_id"}]},
        }

    def test_by_type_lists_all_node_ids_per_type(self):
        idx = _build_indexes(self._sample_nodes())
        self.assertEqual(("f1",), idx["by_type"]["Formula"])
        self.assertEqual(("v1",), idx["by_type"]["ValueMap"])
        self.assertEqual(("a1",), idx["by_type"]["ColumnAlias"])

    def test_by_column_includes_grounding_binding_and_valuemap(self):
        idx = _build_indexes(self._sample_nodes())
        # ValueMap on products.name
        self.assertIn("v1", idx["by_column"][("products", "name")])
        # Formula grounding on products.price
        self.assertIn("f1", idx["by_column"][("products", "price")])
        # ColumnAlias binding on products.product_id
        self.assertIn("a1", idx["by_column"][("products", "product_id")])

    def test_by_name_token_tokenizes_and_drops_stopwords(self):
        idx = _build_indexes(self._sample_nodes())
        # 'the' is a stopword and should not appear as an index key
        self.assertNotIn("the", idx["by_name_token"])
        # 'product' and 'table' from "the product table" should appear (>=2 chars, non-stop)
        self.assertIn("product", idx["by_name_token"])
        self.assertIn("table", idx["by_name_token"])
        # 'average', 'product', 'price' from the Formula name
        self.assertIn("average", idx["by_name_token"])
        # 'in_stock' from the ValueMap value -- NO, values are not tokenized; only names
        # are. So 'stock' should NOT be in the index because it lives in the 'value' field.
        self.assertNotIn("stock", idx["by_name_token"])

    def test_by_column_no_duplicate_ids_when_grounding_has_multiple_entries_per_column(self):
        # A Formula with two grounding entries pointing at the same column
        # (one bare, one value-bearing) must produce ONE entry in by_column,
        # not two -- otherwise downstream consumers double-count.
        nodes = {
            "f1": {"id": "f1", "type": "Formula", "name": "x", "expression": "x",
                   "grounding": [
                       {"term": "label", "table": "t", "column": "c"},
                       {"term": "+", "table": "t", "column": "c", "value": "+"},
                   ]},
        }
        idx = _build_indexes(nodes)
        ids = idx["by_column"][("t", "c")]
        self.assertEqual(len(ids), len(set(ids)),
                         f"duplicate ids in by_column: {ids}")
        self.assertEqual(("f1",), ids)


_SHOP_DDL = _TEST_DDL  # alias for clarity in build tests
_OTHER_DDL = """
CREATE TABLE x (
    id INTEGER PRIMARY KEY,
    label TEXT
);
"""


class BuildGraph(unittest.TestCase):
    def test_empty_inputs_empty_output(self):
        out = build_graph([], {})
        self.assertEqual({}, out)

    def test_single_record_produces_single_db_graph(self):
        out = build_graph([_ITEM_RULE_A], {"shop": _SHOP_DDL})
        self.assertEqual({"shop"}, set(out.keys()))
        g = out["shop"]
        self.assertEqual("shop", g.db_id)
        self.assertEqual(1, len(g.nodes))
        rule = next(iter(g.nodes.values()))
        self.assertEqual("Rule", rule["type"])
        # L2 layer populated from the DDL
        self.assertIn(("products", "price"), g.columns)
        self.assertGreaterEqual(len(g.fk_edges), 1)  # orders.product_id -> products.product_id

    def test_multi_db_returns_per_db_graphs(self):
        item_x = {
            "question_id": 1, "db_id": "other", "difficulty": "simple",
            "question": "", "raw_evidence": "", "notes": "",
            "nodes": [
                {"id": "v1", "type": "ValueMap", "name": "active",
                 "table": "x", "column": "label", "value": "active"},
            ],
        }
        out = build_graph(
            [_ITEM_RULE_A, item_x],
            {"shop": _SHOP_DDL, "other": _OTHER_DDL},
        )
        self.assertEqual({"shop", "other"}, set(out.keys()))

    def test_missing_schema_raises_key_error(self):
        with self.assertRaises(KeyError):
            build_graph([_ITEM_RULE_A], {})


import json as _json
import tempfile as _tempfile


class Serialization(unittest.TestCase):
    def _build(self):
        return build_graph([_ITEM_RULE_A], {"shop": _SHOP_DDL})["shop"]

    def test_round_trip_identity(self):
        g = self._build()
        with _tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            dump_graph(g, path)
            loaded = load_graph(path)
            self.assertEqual(g, loaded)
        finally:
            os.unlink(path)

    def test_dump_writes_expected_top_level_keys(self):
        g = self._build()
        with _tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            dump_graph(g, path)
            with open(path) as fh:
                d = _json.load(fh)
            for k in ("version", "db_id", "match_signature_hash", "nodes", "columns", "fk_edges", "indexes", "provenance"):
                self.assertIn(k, d, f"missing top-level key {k!r}")
            self.assertEqual("p1b.2", d["version"])
            for sub in ("by_type", "by_column", "by_name_token"):
                self.assertIn(sub, d["indexes"])
        finally:
            os.unlink(path)

    def test_load_restores_tuple_types(self):
        g = self._build()
        with _tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            dump_graph(g, path)
            loaded = load_graph(path)
            # fk_edges is declared tuple[tuple[(str,str),(str,str)],...]
            self.assertIsInstance(loaded.fk_edges, tuple)
            for edge in loaded.fk_edges:
                self.assertIsInstance(edge, tuple)
                self.assertIsInstance(edge[0], tuple)
                self.assertIsInstance(edge[1], tuple)
            # by_type values are tuples
            for v in loaded.by_type.values():
                self.assertIsInstance(v, tuple)
        finally:
            os.unlink(path)

    def test_round_trip_preserves_match_signature_hash(self):
        g = self._build()
        self.assertNotEqual("", g.match_signature_hash,
                            "build_graph should set the hash")
        with _tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            dump_graph(g, path)
            loaded = load_graph(path)
            self.assertEqual(g.match_signature_hash, loaded.match_signature_hash)
        finally:
            os.unlink(path)

    def test_load_warns_on_match_signature_hash_mismatch(self):
        import warnings as _warnings
        g = self._build()
        with _tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            dump_graph(g, path)
            # Patch the file to have a different hash to simulate staleness.
            with open(path) as fh:
                d = _json.load(fh)
            d["match_signature_hash"] = "deadbeef" * 8  # 64-char fake
            with open(path, "w") as fh:
                _json.dump(d, fh)
            with _warnings.catch_warnings(record=True) as w:
                _warnings.simplefilter("always")
                load_graph(path)
                self.assertTrue(
                    any("Stale joint graph" in str(warn.message) for warn in w),
                    f"expected staleness warning, got {[str(x.message) for x in w]}",
                )
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
