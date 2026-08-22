#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUT="${1:-$ROOT/reproduced_release_assets}"

"$PYTHON" "$ROOT/code/publication/wfp_render_release_publication_assets.py"   --fig0-dir "$ROOT/results/figure_ready"   --data-dir "$ROOT/results/figure_ready"   --out "$OUT"

echo
echo "Reproduced release assets under: $OUT"
