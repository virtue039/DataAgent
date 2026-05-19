"""CLI entry point for the BIRD evaluation harness.

Setup & run on the Linux server -- see the scripts/ directory:
  bash scripts/setup.sh           # create venv, install dependencies
  bash scripts/build_kb.sh        # build the flat-RAG knowledge base from BIRD train
  bash scripts/smoke_test.sh      # verify data loading, no API calls (free)
  bash scripts/run_oracle_gap.sh  # Research_Plan.md step 2: none / retrieval / oracle

Direct usage:
  python run_eval.py --bird-dir data/bird_dev --setting oracle    --limit 50
  python run_eval.py --bird-dir data/bird_dev --setting retrieval --limit 50
"""
from __future__ import annotations

import argparse

from bird_eval.config import SETTINGS, Config
from bird_eval.runner import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BIRD dev evaluation across the no-knowledge / retrieval / oracle settings."
    )
    parser.add_argument("--bird-dir", required=True,
                        help="Path to the extracted BIRD dev folder.")
    parser.add_argument("--setting", choices=SETTINGS, default="oracle",
                        help="none = no knowledge, oracle = gold evidence, retrieval = flat-RAG.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N questions (omit = full dev set).")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of parallel LLM requests.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--schema-sample-rows", type=int, default=0,
                        help="Include N sample rows per table in the prompt (0 = DDL only).")
    parser.add_argument("--model", default=None,
                        help="Override the model id (default: $DEEPSEEK_MODEL).")
    parser.add_argument("--kb-path", default="data/kb_train.json",
                        help="Knowledge base JSON for the retrieval setting (build with build_kb.py).")
    parser.add_argument("--retrieval-top-k", type=int, default=5,
                        help="Number of knowledge entries to retrieve (retrieval setting).")
    parser.add_argument("--embedding-model", default=None,
                        help="sentence-transformers model for retrieval (default: $EMBEDDING_MODEL).")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build prompts (and run retrieval) but make no LLM API calls.")
    args = parser.parse_args()

    kwargs = dict(
        bird_dir=args.bird_dir,
        setting=args.setting,
        limit=args.limit,
        concurrency=args.concurrency,
        temperature=args.temperature,
        schema_sample_rows=args.schema_sample_rows,
        kb_path=args.kb_path,
        retrieval_top_k=args.retrieval_top_k,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    if args.model:
        kwargs["model"] = args.model
    if args.embedding_model:
        kwargs["embedding_model"] = args.embedding_model
    run(Config(**kwargs))


if __name__ == "__main__":
    main()
