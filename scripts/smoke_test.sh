#!/usr/bin/env bash
# Verify data loading + prompt construction with no API calls (free).
#
# Env overrides:
#   BIRD_DIR  path to the BIRD dev folder  (default: data/bird_dev)
set -euo pipefail
cd "$(dirname "$0")/.."

BIRD_DIR="${BIRD_DIR:-data/bird_dev}"
.venv/bin/python run_eval.py --bird-dir "$BIRD_DIR" --setting oracle --limit 5 --dry-run
