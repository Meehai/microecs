#!/usr/bin/env bash
# docs/build_docs.sh -- build THIS project's docs site with the shared pdoc docs builder
# (docs/shared/ -- the same copy-pastable library every project uses; see docs/shared/README.md).
# Everything project-specific lives in docs/config.py (+ optional docs/custom.css) and the
# prose pages under docs/source/; this file is identical in every project by design.
#
#   bash docs/build_docs.sh            # build into ./public, then print a file:// link
#   bash docs/build_docs.sh public     # build into ./public  (GitLab Pages serves this dir in CI)
set -euo pipefail
cd "$(dirname "$0")/.."                 # repo root (this script lives in docs/)
OUT="${1:-public}"

# one pure-python build dependency; auto-install so a fresh checkout just works
python -c "import pdoc" 2>/dev/null || { echo "[docs] installing pdoc..."; pip install --quiet pdoc; }

rm -rf "$OUT"
python docs/shared/build_docs.py "$OUT"

echo "[docs] built $OUT/  ->  open file://$(cd "$OUT" && pwd)/index.html"
