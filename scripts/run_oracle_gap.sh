#!/usr/bin/env bash
# Research_Plan.md step 2: quantify the knowledge gap on BIRD dev.
# Runs three settings (none / retrieval / oracle) and writes per-run JSON to results/.
#
# Env overrides:
#   BIRD_DIR     path to the BIRD dev folder  (default: data/bird_dev)
#   LIMIT        questions to evaluate        (default: 50; set LIMIT=0 for the full dev set)
#   CONCURRENCY  parallel LLM requests        (default: 8)
set -euo pipefail
cd "$(dirname "$0")/.."

BIRD_DIR="${BIRD_DIR:-data/bird_dev}"
LIMIT="${LIMIT:-50}"
CONCURRENCY="${CONCURRENCY:-8}"

for SETTING in none retrieval oracle; do
  echo
  echo "######## setting: $SETTING ########"
  if [ "$LIMIT" != "0" ]; then
    .venv/bin/python run_eval.py \
      --bird-dir "$BIRD_DIR" --setting "$SETTING" \
      --concurrency "$CONCURRENCY" --limit "$LIMIT"
  else
    .venv/bin/python run_eval.py \
      --bird-dir "$BIRD_DIR" --setting "$SETTING" \
      --concurrency "$CONCURRENCY"
  fi
done

echo
echo "Done. Compare the 'Execution Acc.' lines above; per-run JSON is in results/."
echo "Expected ordering: none < retrieval < oracle. The none->oracle gap is the"
echo "motivation number; retrieval is the flat-RAG baseline the graph method must beat."
