#!/usr/bin/env bash
# Build the flat-RAG knowledge base from BIRD train evidence (non-overlap KB).
#
# Usage:
#   bash scripts/build_kb.sh [path/to/train.json]
# Default train.json location: data/bird_train/train.json
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_JSON="${1:-data/bird_train/train.json}"
.venv/bin/python build_kb.py --train-json "$TRAIN_JSON" --out data/kb_train.json
