#!/usr/bin/env bash
# Sample BIRD dev evidence into a typed-node annotation template (Research_Plan.md step 3).
#
# Env overrides:
#   BIRD_DIR  path to the BIRD dev folder  (default: data/bird_dev)
#   N         number of items to sample    (default: 40)
set -euo pipefail
cd "$(dirname "$0")/.."

BIRD_DIR="${BIRD_DIR:-data/bird_dev}"
N="${N:-40}"

.venv/bin/python sample_evidence.py --bird-dir "$BIRD_DIR" --n "$N" --out annotation/to_annotate.json

echo
echo "Template written to annotation/to_annotate.json (schemas embedded)."
echo "Fill each item's 'nodes' per annotation/node_schema.json, then validate:"
echo "  .venv/bin/python validate_annotations.py annotation/to_annotate.json"
