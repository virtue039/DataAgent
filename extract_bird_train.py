"""CLI entry point for P2 batch extraction over BIRD train.

See docs/superpowers/specs/2026-05-21-p2-bird-train-extraction-design.md.

Direct usage:
  .venv/bin/python extract_bird_train.py             # full 8783-item run
  .venv/bin/python extract_bird_train.py --limit 10  # smoke (resumable)

Output:
  annotation/train_extracted.json          : final v2-shaped output
  annotation/train_extracted.partial.json  : in-flight checkpoint
  annotation/train_extracted.dropped.jsonl : per-drop sidecar
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

from bird_eval.batch_extract import run_batch
from bird_eval.config import Config
from bird_eval.llm import LLMClient
from bird_eval.train_schemas import build_train_schemas


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-extract v2 typed-node annotations from BIRD train evidence."
    )
    parser.add_argument("--train-json",
                        default="data/bird_train/train/train.json",
                        help="BIRD train JSON path.")
    parser.add_argument("--train-dbs-root",
                        default="data/bird_train/train/train_databases",
                        help="Root containing one <db_id>/<db_id>.sqlite per db.")
    parser.add_argument("--schemas-cache",
                        default="data/train_schemas.json",
                        help="Where to read/write the prebuilt schemas dict.")
    parser.add_argument("--output",
                        default="annotation/train_extracted.json",
                        help="Final output path; .partial.json sibling is the checkpoint.")
    parser.add_argument("--sidecar",
                        default="annotation/train_extracted.dropped.jsonl",
                        help="Sidecar JSONL for dropped items.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N items with non-empty evidence (smoke mode).")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="ThreadPool size for parallel LLM calls.")
    parser.add_argument("--checkpoint-every", type=int, default=200,
                        help="Write partial.json after every N completed items.")
    parser.add_argument(
        "--consecutive-llm-error-threshold", type=int, default=100,
        help="Abort if this many LLM errors hit in a row (endpoint down).",
    )
    args = parser.parse_args()

    # Step 1: Load schemas (cached if available).
    print(f"Loading train schemas from {args.train_dbs_root} "
          f"(cache: {args.schemas_cache})...")
    t0 = time.time()
    schemas = build_train_schemas(
        train_dbs_root=Path(args.train_dbs_root),
        cache_path=Path(args.schemas_cache),
    )
    print(f"  -> {len(schemas)} dbs ready in {time.time() - t0:.1f}s")

    # Step 2: Load train items (keep only those with non-empty evidence).
    raw = json.loads(Path(args.train_json).read_text(encoding="utf-8"))
    # BIRD train items don't carry question_id; assign by source order so
    # the skip-set / sidecar references are stable across runs.
    items: list[dict] = []
    for idx, it in enumerate(raw):
        ev = (it.get("evidence") or "").strip()
        if not ev:
            continue
        items.append({
            "question_id": idx,
            "db_id": it["db_id"],
            "difficulty": it.get("difficulty", ""),
            "question": it.get("question", ""),
            "raw_evidence": ev,
            "notes": "",
        })
    if args.limit is not None:
        items = items[: args.limit]
    print(f"  -> {len(items)} items with non-empty evidence "
          f"(limited to {args.limit})" if args.limit else
          f"  -> {len(items)} items with non-empty evidence")

    # Step 3: Build the LLM client (Config picks env vars up via .env).
    cfg = Config(bird_dir=Path("data/bird_dev"), setting="none")
    llm = LLMClient(cfg)

    # Step 4: Dispatch the batch.
    meta = {
        "source": "BIRD train (P2)",
        "extraction_model": cfg.model,
        "spec": "docs/superpowers/specs/2026-05-21-p2-bird-train-extraction-design.md",
    }
    pbar = tqdm(total=len(items), desc="extract", unit="item")
    stats = run_batch(
        items=items, schemas=schemas, llm=llm,
        output_path=Path(args.output),
        sidecar_path=Path(args.sidecar),
        meta=meta,
        concurrency=args.concurrency,
        checkpoint_every=args.checkpoint_every,
        consecutive_llm_error_threshold=args.consecutive_llm_error_threshold,
        progress_cb=pbar.update,
    )
    pbar.close()

    # Step 5: Print the final summary.
    total = stats["total"]
    kept = stats["kept"]
    p = lambda n: f"{n:>5d} ({100 * n / max(total, 1):5.2f}%)"
    print("")
    print(f"Total train items with evidence : {total}")
    print(f"Successfully extracted          : {p(kept)}")
    print(f"Dropped — llm_error             : {p(stats['dropped_llm_error'])}")
    print(f"Dropped — extract_failed        : {p(stats['dropped_extract_failed'])}")
    print(f"Dropped — grounding_hallucinated: {p(stats['dropped_grounding_hallucinated'])}")
    print(f"Dropped — schema_missing       : {p(stats['dropped_schema_missing'])}")
    out_size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"Wrote: {args.output} ({kept} items, {out_size_mb:.1f}MB)")
    sidecar_lines = sum(
        1 for _ in Path(args.sidecar).open()
    ) if Path(args.sidecar).exists() else 0
    print(f"Wrote: {args.sidecar} ({sidecar_lines} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
