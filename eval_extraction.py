"""CLI entry point for the L1 auto-extractor + evaluator (P1a).

Runs `bird_eval.extraction.extract` against each item in the v2 gold standard
file, compares predicted nodes to gold, and saves a JSON report under results/.
Mirrors run_eval.py's shape.

Usage:
  .venv/bin/python eval_extraction.py [--annotation PATH] [--concurrency N]
                                      [--limit N] [--output PATH] [--model NAME]
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# Load .env before importing modules that read env at import time.
load_dotenv(Path(__file__).resolve().parent / ".env")

from bird_eval.config import Config
from bird_eval.extraction import extract
from bird_eval.extraction_eval import aggregate, compare_items
from bird_eval.llm import LLMClient


def _process(item: dict, ddl: str, llm: LLMClient) -> dict:
    try:
        predicted, sanitize_stats = extract(item["raw_evidence"], item["db_id"], ddl, llm)
        err = None
    except Exception as e:  # noqa: BLE001 - surface any extraction error per-item
        predicted = []
        sanitize_stats = {"dropped_groundings": 0, "dropped_nodes": 0}
        err = f"extract failed: {e}"
    cmp = compare_items(predicted, item["nodes"])
    return {
        "question_id": item["question_id"],
        "db_id": item["db_id"],
        "predicted": predicted,
        "gold": item["nodes"],
        "compare": cmp,
        "sanitize_stats": sanitize_stats,
        "error": err,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the L1 auto-extractor against the v2 gold standard."
    )
    parser.add_argument(
        "--annotation", default="annotation/to_annotate.json",
        help="Path to the v2 gold-standard file.",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N items (omit = all).")
    parser.add_argument("--output", default=None,
                        help="Output report path. Default: results/extraction_eval_<ts>.json")
    parser.add_argument("--model", default=None,
                        help="Override DEEPSEEK_MODEL.")
    args = parser.parse_args()

    data = json.loads(Path(args.annotation).read_text(encoding="utf-8"))
    items = data["items"]
    schemas = data["schemas"]
    if args.limit is not None:
        items = items[: args.limit]

    cfg_kwargs = {"bird_dir": Path("data/bird_dev")}  # bird_dir is unused but Config requires it
    if args.model:
        cfg_kwargs["model"] = args.model
    cfg = Config(**cfg_kwargs)
    llm = LLMClient(cfg)

    def worker(it: dict) -> dict:
        return _process(it, schemas[it["db_id"]], llm)

    print(
        f"Loaded {len(items)} items from {args.annotation} | "
        f"model={cfg.model} | concurrency={args.concurrency}"
    )
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for rec in tqdm(pool.map(worker, items), total=len(items), desc="extract"):
            results.append(rec)

    summary = aggregate([r["compare"] for r in results])
    summary["n_items_returning_empty"] = sum(1 for r in results if not r["predicted"])
    summary["n_errors"] = sum(1 for r in results if r["error"])
    summary["model"] = cfg.model

    print("\n" + "=" * 50)
    print(f"  Model           : {summary['model']}")
    print(f"  Items           : {summary['n_items']}")
    print(f"  Empty predicted : {summary['n_items_returning_empty']}")
    print(f"  Errors          : {summary['n_errors']}")
    o = summary["overall"]
    print(f"  Overall  P/R/F1 : {o['precision']:.3f} / {o['recall']:.3f} / {o['f1']:.3f}")
    print(f"  Grounding acc.  : {summary['grounding_acc_on_matched']:.3f}")
    for t, m in summary["by_type"].items():
        print(
            f"    - {t:<12}: F1 {m['f1']:.3f}  "
            f"(P {m['precision']:.3f}  R {m['recall']:.3f}  support {m['support']})"
        )
    print("=" * 50)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else Path("results") / f"extraction_eval_{cfg.model.replace('/', '-')}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "config": {
            "annotation": args.annotation,
            "model": cfg.model,
            "concurrency": args.concurrency,
            "limit": args.limit,
        },
        "per_item": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
