"""CLI entry point for P4-revised: batch extraction over BIRD DEV.

See docs/superpowers/specs/2026-05-22-p4r-within-db-cv-design.md.

Direct usage:
  .venv/bin/python scripts/extract_bird_dev.py           # full 1534-item run
  .venv/bin/python scripts/extract_bird_dev.py --limit 5 # smoke

Output:
  annotation/dev_extracted.json          : final v2-shaped output
  annotation/dev_extracted.partial.json  : in-flight checkpoint
  annotation/dev_extracted.dropped.jsonl : per-drop sidecar
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bird_eval.batch_extract import run_batch  # noqa: E402
from bird_eval.config import Config  # noqa: E402
from bird_eval.data import get_schema_ddl, locate_bird_files  # noqa: E402
from bird_eval.llm import LLMClient  # noqa: E402


def _build_dev_schemas(bird_dir: Path) -> dict[str, str]:
    """Walk dev databases dir, return {db_id: ddl_str} for 11 dev dbs."""
    _, db_root = locate_bird_files(bird_dir)
    schemas: dict[str, str] = {}
    for db_dir in sorted(p for p in db_root.iterdir() if p.is_dir()):
        sqlite_path = db_dir / f"{db_dir.name}.sqlite"
        if not sqlite_path.exists():
            continue
        schemas[db_dir.name] = get_schema_ddl(sqlite_path)
    return schemas


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-extract v2 typed-node annotations from BIRD DEV evidence."
    )
    parser.add_argument("--bird-dir", default="data/bird_dev",
                        help="Path to the extracted BIRD dev folder.")
    parser.add_argument("--output", default="annotation/dev_extracted.json")
    parser.add_argument("--sidecar", default="annotation/dev_extracted.dropped.jsonl")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N items with non-empty evidence.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--consecutive-llm-error-threshold", type=int, default=100)
    args = parser.parse_args()

    # Step 1: Build dev schemas (11 dbs; no caching script — fast enough).
    print(f"Loading dev schemas from {args.bird_dir}...")
    t0 = time.time()
    schemas = _build_dev_schemas(Path(args.bird_dir))
    print(f"  -> {len(schemas)} dbs ready in {time.time() - t0:.1f}s")

    # Step 2: Load dev items.
    dev_json, _ = locate_bird_files(Path(args.bird_dir))
    raw = json.loads(dev_json.read_text(encoding="utf-8"))
    items: list[dict] = []
    for idx, it in enumerate(raw):
        ev = (it.get("evidence") or "").strip()
        if not ev:
            continue
        items.append({
            "question_id": it.get("question_id", idx),
            "db_id": it["db_id"],
            "difficulty": it.get("difficulty", ""),
            "question": it.get("question", ""),
            "raw_evidence": ev,
            "notes": "",
        })
    if args.limit is not None:
        items = items[: args.limit]
    print(f"  -> {len(items)} items with non-empty evidence")

    # Step 3: LLM client.
    cfg = Config(bird_dir=Path(args.bird_dir), setting="none")
    llm = LLMClient(cfg)

    # Step 4: Batch.
    meta = {
        "source": "BIRD dev (P4-revised)",
        "extraction_model": cfg.model,
        "spec": "docs/superpowers/specs/2026-05-22-p4r-within-db-cv-design.md",
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

    total = stats["total"]
    kept = stats["kept"]
    p = lambda n: f"{n:>5d} ({100 * n / max(total, 1):5.2f}%)"
    print("")
    print(f"Total dev items with evidence  : {total}")
    print(f"Successfully extracted          : {p(kept)}")
    print(f"Dropped — llm_error             : {p(stats['dropped_llm_error'])}")
    print(f"Dropped — extract_failed        : {p(stats['dropped_extract_failed'])}")
    print(f"Dropped — grounding_hallucinated: {p(stats['dropped_grounding_hallucinated'])}")
    print(f"Dropped — schema_missing       : {p(stats['dropped_schema_missing'])}")
    out_size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"Wrote: {args.output} ({kept} items, {out_size_mb:.1f}MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
