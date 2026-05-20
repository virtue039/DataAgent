"""Unit tests for annotation/migrate_v1_to_v2.py.

Synthetic per-item fixtures stand in for full BIRD records to keep the tests
hermetic. Each test checks one rewrite pattern from the design spec §6.

Run with:
  .venv/bin/python -m unittest tests.test_migration -v
"""
from __future__ import annotations

import copy
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "annotation"))

from migrate_v1_to_v2 import migrate_item  # noqa: E402


def _wrap(qid, db_id, nodes, notes=""):
    return {
        "question_id": qid,
        "db_id": db_id,
        "difficulty": "simple",
        "question": "",
        "raw_evidence": "",
        "nodes": nodes,
        "notes": notes,
    }


class ColumnAliasMigration(unittest.TestCase):
    """v1 Concept-with-definition (column rename) -> v2 ColumnAlias (single binding)."""

    def test_qid_491_card_games_mcm_name(self):
        item = _wrap(491, "card_games", [
            {"id": "c1", "type": "Concept", "name": "magic card market name",
             "definition": "sets.mcmName (the MCM-system name for the set)"},
        ])
        out = migrate_item(copy.deepcopy(item))
        self.assertEqual(len(out["nodes"]), 1)
        n = out["nodes"][0]
        self.assertEqual(n["type"], "ColumnAlias")
        self.assertEqual(n["name"], "magic card market name")
        self.assertEqual(n["bindings"], [{"table": "sets", "column": "mcmName"}])
        self.assertNotIn("definition", n)

    def test_qid_122_financial_two_aliases(self):
        item = _wrap(122, "financial", [
            {"id": "c1", "type": "Concept", "name": "district name", "definition": "district.A2"},
            {"id": "c2", "type": "Concept", "name": "region name", "definition": "district.A3"},
        ])
        out = migrate_item(copy.deepcopy(item))
        names = {(n["name"], tuple(sorted((b["table"], b["column"]) for b in n["bindings"])))
                 for n in out["nodes"]}
        self.assertIn(("district name", (("district", "A2"),)), names)
        self.assertIn(("region name", (("district", "A3"),)), names)
        self.assertTrue(all(n["type"] == "ColumnAlias" for n in out["nodes"]))


class MultiBindingColumnAlias(unittest.TestCase):
    """v1 Concept-with-definition (column tuple) -> v2 ColumnAlias with multiple bindings."""

    def test_qid_978_location_coordinates(self):
        item = _wrap(978, "formula_1", [
            {"id": "c1", "type": "Concept", "name": "location coordinates",
             "definition": "(circuits.lat, circuits.lng) -- pair of columns projected together"},
            {"id": "v1", "type": "ValueMap", "name": "Austria",
             "table": "circuits", "column": "country", "value": "Austria"},
        ])
        out = migrate_item(copy.deepcopy(item))
        a = next(n for n in out["nodes"] if n["type"] == "ColumnAlias")
        self.assertEqual(a["name"], "location coordinates")
        self.assertEqual(a["bindings"],
                         [{"table": "circuits", "column": "lat"},
                          {"table": "circuits", "column": "lng"}])
        # ValueMap sibling preserved unchanged.
        self.assertTrue(any(n["type"] == "ValueMap" and n["name"] == "Austria"
                            for n in out["nodes"]))


class GroundingValueAddition(unittest.TestCase):
    """v1 Formula with value-literal in expression -> v2 Formula with grounding.value entries.
    ValueMap siblings are preserved (double-encoding accepted)."""

    def test_qid_317_toxicology_pct_carcinogenic(self):
        item = _wrap(317, "toxicology", [
            {"id": "v1", "type": "ValueMap", "name": "carcinogenic molecule",
             "table": "molecule", "column": "label", "value": "+"},
            {"id": "v2", "type": "ValueMap", "name": "Chlorine element",
             "table": "atom", "column": "element", "value": "cl"},
            {"id": "f1", "type": "Formula",
             "name": "percentage of carcinogenic molecules containing Chlorine",
             "expression": "SUM(label = '+' AND element = 'cl') / COUNT(molecule_id) * 100",
             "grounding": [
                 {"term": "label", "table": "molecule", "column": "label"},
                 {"term": "element", "table": "atom", "column": "element"},
                 {"term": "molecule_id", "table": "molecule", "column": "molecule_id"}]},
        ])
        out = migrate_item(copy.deepcopy(item))
        # both ValueMaps still present
        self.assertEqual(sum(1 for n in out["nodes"] if n["type"] == "ValueMap"), 2)
        # Formula now has value-bearing grounding entries
        f1 = next(n for n in out["nodes"] if n["id"] == "f1")
        gs = f1["grounding"]
        self.assertTrue(any(g.get("value") == "+" and g["column"] == "label" for g in gs),
                        f"missing label='+' grounding; got {gs}")
        self.assertTrue(any(g.get("value") == "cl" and g["column"] == "element" for g in gs),
                        f"missing element='cl' grounding; got {gs}")

    def test_qid_352_card_games_chinese_simplified(self):
        item = _wrap(352, "card_games", [
            {"id": "v1", "type": "ValueMap", "name": "Chinese Simplified",
             "table": "foreign_data", "column": "language", "value": "Chinese Simplified"},
            {"id": "f1", "type": "Formula",
             "name": "percentage of cards in Chinese Simplified",
             "expression": "SUM(id WHERE language = 'Chinese Simplified') / COUNT(id) * 100",
             "grounding": [
                 {"term": "id", "table": "foreign_data", "column": "id"},
                 {"term": "language", "table": "foreign_data", "column": "language"}]},
        ])
        out = migrate_item(copy.deepcopy(item))
        # ValueMap sibling preserved.
        self.assertTrue(any(n["type"] == "ValueMap" and n["name"] == "Chinese Simplified"
                            for n in out["nodes"]))
        f1 = next(n for n in out["nodes"] if n["id"] == "f1")
        self.assertTrue(
            any(g.get("value") == "Chinese Simplified" and g["column"] == "language"
                for g in f1["grounding"]),
            f"missing language='Chinese Simplified' grounding; got {f1['grounding']}",
        )

    def test_qid_1337_student_club_october_speaker(self):
        item = _wrap(1337, "student_club", [
            {"id": "f1", "type": "Formula", "name": "total budgeted amount for an event",
             "expression": "SUM(amount) WHERE event_name = 'October Speaker'",
             "grounding": [
                 {"term": "amount", "table": "budget", "column": "amount"},
                 {"term": "event_name", "table": "event", "column": "event_name"}]},
            {"id": "v1", "type": "ValueMap", "name": "October Speaker event",
             "table": "event", "column": "event_name", "value": "October Speaker"},
        ])
        out = migrate_item(copy.deepcopy(item))
        self.assertTrue(any(n["type"] == "ValueMap" and n["name"] == "October Speaker event"
                            for n in out["nodes"]))
        f1 = next(n for n in out["nodes"] if n["id"] == "f1")
        self.assertTrue(
            any(g.get("value") == "October Speaker" and g["column"] == "event_name"
                for g in f1["grounding"]),
            f"missing event_name='October Speaker' grounding; got {f1['grounding']}",
        )


class DependsOnAddition(unittest.TestCase):
    """v1 Formula that the notes flagged as depending on a Rule -> v2 Formula.depends_on."""

    def test_qid_954_race_completion(self):
        item = _wrap(954, "formula_1", [
            {"id": "r1", "type": "Rule", "name": "year between 2007 and 2009",
             "condition": "year BETWEEN 2007 AND 2009",
             "grounding": [{"term": "year", "table": "races", "column": "year"}]},
            {"id": "c1", "type": "Concept", "name": "race completion", "defined_by": "r2"},
            {"id": "r2", "type": "Rule", "name": "finished the race (time not null)",
             "condition": "time IS NOT NULL",
             "grounding": [{"term": "time", "table": "results", "column": "time"}]},
            {"id": "f1", "type": "Formula", "name": "race completion percentage",
             "expression": "COUNT(DriverID WHERE time IS NOT NULL AND year BETWEEN 2007 AND 2009) / COUNT(DriverID WHERE year BETWEEN 2007 AND 2009) * 100",
             "grounding": [
                 {"term": "DriverID", "table": "results", "column": "driverId"},
                 {"term": "time", "table": "results", "column": "time"},
                 {"term": "year", "table": "races", "column": "year"}]},
        ])
        out = migrate_item(copy.deepcopy(item))
        f1 = next(n for n in out["nodes"] if n["id"] == "f1")
        self.assertEqual(set(f1.get("depends_on", [])), {"r1", "r2"})


class Idempotency(unittest.TestCase):
    def test_migrating_v2_item_is_a_noop(self):
        # Already-v2 item: should come out unchanged.
        item = _wrap(491, "card_games", [
            {"id": "a1", "type": "ColumnAlias", "name": "magic card market name",
             "bindings": [{"table": "sets", "column": "mcmName"}]},
        ])
        before = copy.deepcopy(item)
        out = migrate_item(item)
        self.assertEqual(before, out)

    def test_unaffected_item_unchanged(self):
        # qid that's not in the spec §6 table -- migrate leaves it alone.
        item = _wrap(519, "card_games", [
            {"id": "v1", "type": "ValueMap", "name": "Battlebond set",
             "table": "sets", "column": "name", "value": "Battlebond"},
        ])
        before = copy.deepcopy(item)
        out = migrate_item(item)
        self.assertEqual(before, out)


if __name__ == "__main__":
    unittest.main()
