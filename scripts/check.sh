#!/usr/bin/env bash
# Compile-check every Python source. Needs only a Python 3 interpreter, no deps.
# Run this first to catch syntax errors before installing anything.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
"$PY" -m py_compile bird_eval/*.py run_eval.py build_kb.py sample_evidence.py validate_annotations.py
echo "OK: all Python files compile."
