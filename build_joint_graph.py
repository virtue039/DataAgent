"""CLI entry point for the joint-graph builder (P1b).

Reads the v2 gold standard at annotation/to_annotate.json (or another v2
annotation file), builds one JointGraph per db_id, writes JSON files to
the output directory.

Usage:
  .venv/bin/python build_joint_graph.py [--annotation PATH] [--output-dir PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bird_eval.joint_graph import build_graph, dump_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-DB JointGraphs from a v2 annotation file."
    )
    parser.add_argument(
        "--annotation", default="annotation/to_annotate.json",
        help="Path to v2 annotation file (default: annotation/to_annotate.json).",
    )
    parser.add_argument(
        "--output-dir", default="data/joint_graphs",
        help="Where to write <db_id>.json files (default: data/joint_graphs).",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.annotation).read_text(encoding="utf-8"))
    items = data["items"]
    schemas = data["schemas"]

    graphs = build_graph(items, schemas)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Built {len(graphs)} graphs from {args.annotation}.")
    total_nodes = 0
    for db_id, g in sorted(graphs.items()):
        out_path = out_root / f"{db_id}.json"
        dump_graph(g, out_path)
        print(
            f"  {db_id:<28} nodes={len(g.nodes):>3}  "
            f"cols={len(g.columns):>3}  fks={len(g.fk_edges):>3}  "
            f"-> {out_path}"
        )
        total_nodes += len(g.nodes)
    print(f"\nTotal L1 nodes across all graphs: {total_nodes}")


if __name__ == "__main__":
    main()
