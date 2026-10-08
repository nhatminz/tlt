#!/usr/bin/env bash
# Recreate the official TLT fork + local Reflex patch; never overwrite a dirty tree.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMIT=bce3df7a4d46473912e9b81bf47bca419729557f
UPSTREAM_ROOT="${UPSTREAM_ROOT:-$ROOT/upstream}"
DEFAULT_SOURCE=https://github.com/mit-han-lab/fastrl
if [[ -f "$ROOT/artifacts/fastrl-bce3df7.bundle" ]];then DEFAULT_SOURCE="$ROOT/artifacts/fastrl-bce3df7.bundle";fi
SOURCE="${FASTRL_GIT_SOURCE:-$DEFAULT_SOURCE}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PATCH="$ROOT/patches/fastrl_reflex.patch"
[[ -f "$PATCH" ]] || { echo 'ERROR: tracked patches/fastrl_reflex.patch missing' >&2; exit 2; }
command -v git >/dev/null || { echo 'ERROR: git is required for bootstrap' >&2; exit 2; }
export GIT_TERMINAL_PROMPT=0
mkdir -p "$UPSTREAM_ROOT"
# A zip/previous checkout may have no nested Git metadata. Migrate only the
# exact recognized legacy source. Preserve it as a reversible backup.
if [[ -d "$UPSTREAM_ROOT/fastrl" ]];then
  if [[ ! -d "$UPSTREAM_ROOT/fastrl/.git" ]] || ! git -C "$UPSTREAM_ROOT/fastrl" apply --reverse --check "$PATCH" >/dev/null 2>&1;then
    if "$PYTHON_BIN" "$ROOT/scripts/upgrade_upstream.py" "$UPSTREAM_ROOT/fastrl";then
      backup="$UPSTREAM_ROOT/../upstream.legacy.opd_upgrade.$(date -u +%Y%m%dT%H%M%S_%N)"
      mv "$UPSTREAM_ROOT/fastrl" "$backup"
      echo "Recognized previous runtime saved at $backup"
    fi
  fi
fi
if [[ ! -e "$UPSTREAM_ROOT/fastrl" ]];then
  stage="$(mktemp -d "$UPSTREAM_ROOT/.fastrl-bootstrap.XXXXXX")"
  echo "Cloning official FastRL (or explicit offline git bundle): $SOURCE"
  git clone --quiet --no-checkout "$SOURCE" "$stage/fastrl" || {
    echo 'ERROR: clone failed; offline server must set FASTRL_GIT_SOURCE to a prepared git bundle/mirror. No fallback to an unverified source.' >&2; exit 2;
  }
  git -C "$stage/fastrl" checkout --quiet --detach "$COMMIT"
  [[ "$(git -C "$stage/fastrl" rev-parse HEAD)" == "$COMMIT" ]] || { echo 'ERROR: upstream commit mismatch' >&2; exit 2; }
  cp -a "$stage/fastrl/third-party/sglang/python" "$stage/pristine_sglang_python"
  git -C "$stage/fastrl" apply --check "$PATCH"
  git -C "$stage/fastrl" apply "$PATCH"
  git -C "$stage/fastrl" apply --reverse --check "$PATCH"
  if [[ -e "$UPSTREAM_ROOT/pristine_sglang_python" ]];then
    diff -qr --exclude=__pycache__ "$stage/pristine_sglang_python" "$UPSTREAM_ROOT/pristine_sglang_python" >/dev/null || {
      echo 'ERROR: pristine source differs from verified pin; existing source was preserved' >&2;exit 2;
    }
  else
    mv "$stage/pristine_sglang_python" "$UPSTREAM_ROOT/pristine_sglang_python"
  fi
  mv "$stage/fastrl" "$UPSTREAM_ROOT/fastrl"
else
  [[ -d "$UPSTREAM_ROOT/fastrl/.git" ]] || {
    echo 'ERROR: existing fastrl has no git identity. Move legacy upstream aside, then bootstrap; nothing was overwritten.' >&2; exit 2;
  }
  [[ "$(git -C "$UPSTREAM_ROOT/fastrl" rev-parse HEAD)" == "$COMMIT" ]] || { echo "ERROR: upstream HEAD must be $COMMIT; refusing to checkout/reset an existing tree" >&2; exit 2; }
  git -C "$UPSTREAM_ROOT/fastrl" apply --reverse --check "$PATCH" || {
    echo 'ERROR: existing upstream does not match applied Reflex patch; use a new UPSTREAM_ROOT or move it aside. No reset/overwrite.' >&2; exit 2;
  }
  if [[ ! -d "$UPSTREAM_ROOT/pristine_sglang_python" ]];then
    stage="$(mktemp -d "$UPSTREAM_ROOT/.pristine-bootstrap.XXXXXX")"
    git -C "$UPSTREAM_ROOT/fastrl" archive "$COMMIT" third-party/sglang/python | tar -x -C "$stage" --strip-components=4
    mv "$stage" "$UPSTREAM_ROOT/pristine_sglang_python"
  fi
fi
"$PYTHON_BIN" "$ROOT/scripts/audit_upstream.py" --upstream-root "$UPSTREAM_ROOT" --require-git
echo "Bootstrap verified: FastRL $COMMIT + exact Reflex patch"
