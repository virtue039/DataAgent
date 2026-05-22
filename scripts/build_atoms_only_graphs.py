"""P5-D3 KB ablation: build "atoms-only" joint graphs by stripping composites.

For each `<db>.json` in --input-dir, applies
`bird_eval.composition_filter.strip_composites` and writes the stripped
graph to --output-dir/<db>.json. Prints per-db node-count deltas.

Usage:
  .venv/bin/python scripts/build_atoms_only_graphs.py \\
      --input-dir data/joint_graphs_dev_cv \\
      --output-dir data/joint_graphs_dev_cv_atoms_only

See docs/superpowers/specs/2026-05-22-p5-d3-composition-specialty-design.md
§3.2 for the ablation rationale; this is the joint-side input to the
D3 eval (joint vs flat retrieval on the same composite-stripped KB).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bird_eval.composition_filter import strip_composites  # noqa: E402
from bird_eval.joint_graph import dump_graph, load_graph  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build composite-stripped (atoms-only) joint graphs.",
    )
    parser.add_argument("--input-dir", default="data/joint_graphs_dev_cv",
                        help="directory of per-db joint-graph JSONs to strip")
    parser.add_argument("--output-dir",
                        default="data/joint_graphs_dev_cv_atoms_only",
                        help="output directory for stripped graphs")
    args = parser.parse_args()

    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    inputs = sorted(in_root.glob("*.json"))
    if not inputs:
        print(f"No *.json files found in {in_root}.", file=sys.stderr)
        return 1

    total_before = 0
    total_after = 0
    print(f"Stripping composites from {len(inputs)} graphs in {in_root}:")
    for src in inputs:
        graph = load_graph(src)
        n_before = len(graph.nodes)
        stripped = strip_composites(graph)
        n_after = len(stripped.nodes)
        delta = n_before - n_after
        out_path = out_root / src.name
        dump_graph(stripped, out_path)
        print(
            f"  {src.stem:<28} nodes: {n_before:>3} -> {n_after:>3} "
            f"(removed {delta:>2})  -> {out_path}"
        )
        total_before += n_before
        total_after += n_after

    print(
        f"\nTotal nodes: {total_before} -> {total_after} "
        f"(removed {total_before - total_after} composites)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
