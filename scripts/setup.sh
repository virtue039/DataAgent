#!/usr/bin/env bash
# One-time environment setup. Run once on the Linux server.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo
echo "Environment ready. Next steps:"
echo "  1. Place the BIRD dev set under   data/bird_dev    (must contain dev.json + the *.sqlite databases)"
echo "  2. Place BIRD train.json under    data/bird_train  (only train.json is needed, for the KB)"
echo "  3. cp .env.example .env  and fill in DEEPSEEK_API_KEY"
echo "  4. bash scripts/build_kb.sh"
echo "  5. bash scripts/smoke_test.sh      # free, no API calls"
echo "  6. bash scripts/run_oracle_gap.sh  # the step-2 experiment"
