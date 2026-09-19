#!/usr/bin/env bash
# Set up an isolated venv (once), install the competitor libs + microecs, then run the full
# workload x library x N sweep. Extra args pass through to run_benchmark.py (e.g. a custom N list).
#
#   ./run_benchmark.sh                 # full matrix + columnar tail -> results.json + tables
#   ./run_benchmark.sh 200 1000 5000   # custom N list (main matrix only)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

if [ ! -d .venv ]; then
    echo "== creating .venv and installing deps =="
    "$PY" -m venv .venv
    ./.venv/bin/pip install -q --upgrade pip
    ./.venv/bin/pip install -q -r requirements.txt
    ./.venv/bin/pip install -q -e ../../          # microecs itself, from the repo root
fi

exec ./.venv/bin/python run_benchmark.py "$@"
