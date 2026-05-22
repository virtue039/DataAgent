"""Build joint graphs from the KB-half of the CV split.

Filters `annotation/dev_extracted.json` to KB qids only, then calls
`bird_eval.joint_graph.build_graph` to produce per-db graphs.

Usage:
  .venv/bin/python scripts/build_cv_joint_graphs.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bird_eval.joint_graph import build_graph, dump_graph  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CV joint graphs from KB half.")
    parser.add_argument("--annotation", default="annotation/dev_extracted.json")
    parser.add_argument("--split", default="data/dev_cv_split.json")
    parser.add_argument("--output-dir", default="data/joint_graphs_dev_cv")
    args = parser.parse_args()

    data = json.loads(Path(args.annotation).read_text(encoding="utf-8"))
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    kb_qids = set(split["kb_qids"])
    schemas = data["schemas"]

    # Filter items to KB qids only.
    kb_items = [it for it in data["items"] if it["question_id"] in kb_qids]
    print(f"Filtered {len(data['items'])} items down to {len(kb_items)} KB items.")

    graphs = build_graph(kb_items, schemas)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Built {len(graphs)} graphs from KB half:")
    total_nodes = 0
    for db_id, g in sorted(graphs.items()):
        out_path = out_root / f"{db_id}.json"
        dump_graph(g, out_path)
        print(f"  {db_id:<28} nodes={len(g.nodes):>3}  cols={len(g.columns):>3}  "
              f"fks={len(g.fk_edges):>3}  -> {out_path}")
        total_nodes += len(g.nodes)
    print(f"\nTotal L1 nodes across all KB graphs: {total_nodes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
