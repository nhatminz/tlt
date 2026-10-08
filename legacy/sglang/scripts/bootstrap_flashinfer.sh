#!/usr/bin/env bash
# Verified source + tracked minimal ABI adapter; no implicit pip installs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-$ROOT/upstream}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PATCH="$ROOT/patches/flashinfer_stable_ffi.patch"
mkdir -p "$UPSTREAM_ROOT"
if [[ ! -d "$UPSTREAM_ROOT/flashinfer" ]];then
  stage="$(mktemp -d "$UPSTREAM_ROOT/.flashinfer-bootstrap.XXXXXX")"
  archive="${FLASHINFER_ARCHIVE:-$stage/source.tar.gz}"
  if [[ ! -f "$archive" ]];then
    [[ "${OFFLINE:-0}" != 1 ]] || { echo 'ERROR: offline bootstrap needs FLASHINFER_ARCHIVE pointing to prepared official0.4.0 sdist' >&2;exit 2; }
    curl --fail --location --retry 2 --max-time 180 \
      https://files.pythonhosted.org/packages/08/29/f5609be182174e8c97124baeb90bb955fe05e2e1353776f48e226c153214/flashinfer_python-0.4.0.tar.gz \
      --output "$archive"
  fi
  expected=c6e4ba1dc1300e17eb8a15c028bc1d79bd7416c9895d32e021430b47674c6c41
  [[ "$(sha256sum "$archive" | cut -d' ' -f1)" == "$expected" ]] || { echo 'ERROR: FlashInfer source SHA256 mismatch' >&2;exit 2; }
  mkdir "$stage/source"
  tar -xf "$archive" -C "$stage/source" --strip-components=1
  # `git -C ignored/subdirectory apply` can silently skip ALL files if it
  # finds an unrelated parent .git. Use patch with an explicit source directory.
  patch --batch --dry-run -p1 -d "$stage/source" < "$PATCH"
  patch --batch -p1 -d "$stage/source" < "$PATCH"
  patch --batch --dry-run --reverse -p1 -d "$stage/source" < "$PATCH"
  mv "$stage/source" "$UPSTREAM_ROOT/flashinfer"
fi
"$PYTHON_BIN" "$ROOT/scripts/audit_upstream.py" --upstream-root "$UPSTREAM_ROOT" --require-flashinfer
