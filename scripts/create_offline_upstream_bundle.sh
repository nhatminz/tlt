#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$ROOT/scripts/bootstrap_upstream.sh"
BUNDLE="${BUNDLE:-$ROOT/artifacts/fastrl-bce3df7.bundle}"
[[ ! -e "$BUNDLE" ]] || { echo 'ERROR: choose a new BUNDLE; refusing overwrite' >&2;exit 2; }
mkdir -p "$(dirname "$BUNDLE")"
git -C "$ROOT/upstream/fastrl" bundle create "$BUNDLE" HEAD
git -C "$ROOT/upstream/fastrl" bundle verify "$BUNDLE"
printf 'Copy to offline server, then run: FASTRL_GIT_SOURCE=%q bash scripts/bootstrap_upstream.sh\n' "$BUNDLE"
