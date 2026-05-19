"""Sample BIRD dev evidence items into an annotation template.

Research_Plan.md step 3: pilot-annotate ~30-50 evidence items into the typed
node schema (annotation/node_schema.json) to validate the L1 graph design.

Sampling is round-robin across databases (with a fixed seed) so the sample
spans many databases rather than clustering in a few. The output is
self-contained: it embeds the schema DDL of every database involved, so the
annotator can ground formula/rule terms to physical columns.

Usage:
  python sample_evidence.py --bird-dir data/bird_dev --n 40 --out annotation/to_annotate.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from bird_eval.data import db_path, get_schema_ddl, load_examples, locate_bird_files


def sample(examples: list, n: int, seed: int) -> list:
    """Round-robin pick across databases for broad coverage."""
    by_db: dict[str, list] = defaultdict(list)
    for e in examples:
        by_db[e.db_id].append(e)

    rng = random.Random(seed)
    for lst in by_db.values():
        rng.shuffle(lst)
    db_order = sorted(by_db)
    rng.shuffle(db_order)

    picked: list = []
    exhausted: set[str] = set()
    i = 0
    while len(picked) < n and len(exhausted) < len(db_order):
        db = db_order[i % len(db_order)]
        if by_db[db]:
            picked.append(by_db[db].pop())
        else:
            exhausted.add(db)
        i += 1
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample BIRD dev evidence into a typed-node annotation template."
    )
    parser.add_argument("--bird-dir", required=True, help="Path to the extracted BIRD dev folder.")
    parser.add_argument("--n", type=int, default=40, help="Number of evidence items to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (reproducible).")
    parser.add_argument("--out", default="annotation/to_annotate.json", help="Output template path.")
    args = parser.parse_args()

    dev_json, db_root = locate_bird_files(args.bird_dir)
    examples = [e for e in load_examples(dev_json) if e.evidence.strip()]
    if not examples:
        raise SystemExit("No examples with non-empty evidence found in the BIRD dev set.")

    picked = sample(examples, args.n, args.seed)
    picked.sort(key=lambda e: (e.db_id, e.question_id))

    schemas: dict[str, str] = {}
    for e in picked:
        if e.db_id not in schemas:
            schemas[e.db_id] = get_schema_ddl(db_path(db_root, e.db_id))

    items = [
        {
            "question_id": e.question_id,
            "db_id": e.db_id,
            "difficulty": e.difficulty,
            "question": e.question,
            "raw_evidence": e.evidence,
            "nodes": [],
            "notes": "",
        }
        for e in picked
    ]

    payload = {
        "meta": {
            "n_items": len(items),
            "n_databases": len(schemas),
            "bird_dir": str(args.bird_dir),
            "seed": args.seed,
            "created": datetime.now().isoformat(timespec="seconds"),
            "schema_ref": "annotation/node_schema.json",
        },
        "schemas": schemas,
        "items": items,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Sampled {len(items)} evidence items from {len(schemas)} databases -> {out}")
    print("Next: fill each item's 'nodes' per annotation/node_schema.json, then run:")
    print(f"  python validate_annotations.py {out}")


if __name__ == "__main__":
    main()
