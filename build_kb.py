"""Build the flat-RAG knowledge base from BIRD train evidence.

The KB is built from BIRD *train* only, so it never overlaps with dev
databases -- the non-overlap setting from Research_Plan.md (OQ2).

Usage:
  python build_kb.py --train-json data/bird_train/train.json --out data/kb_train.json
"""
from __future__ import annotations

import argparse

from bird_eval.knowledge_base import build_kb_from_bird, save_kb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the flat-RAG knowledge base from BIRD train evidence."
    )
    parser.add_argument("--train-json", required=True, help="Path to BIRD train.json.")
    parser.add_argument("--out", default="data/kb_train.json", help="Output KB JSON path.")
    args = parser.parse_args()

    entries = build_kb_from_bird(args.train_json)
    save_kb(entries, args.out)
    print(f"Built KB with {len(entries)} unique knowledge entries -> {args.out}")


if __name__ == "__main__":
    main()
