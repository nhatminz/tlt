"""A clean checkout has no vendored runtime; recreate pinned source for tests.

Only git/source bootstrap, never pip-install or download models in pytest.
Offline runs can supply FASTRL_GIT_SOURCE=/path/to/prepared.bundle.
"""
import os
from pathlib import Path
import subprocess
import sys
import pytest


def pytest_sessionstart(session):
    root=Path(__file__).resolve().parents[1]
    required=[root/'upstream/fastrl/verl/utils/reward_score/__init__.py',
              root/'upstream/pristine_sglang_python/sglang/srt/speculative/eagle_worker.py']
    if not all(p.is_file() for p in required):
        env=os.environ.copy();env.pop('UPSTREAM_ROOT',None)
        env['PYTHON_BIN']=sys.executable
        try:subprocess.run(['bash',str(root/'scripts/bootstrap_upstream.sh')],env=env,check=True)
        except subprocess.CalledProcessError as e:
            raise pytest.UsageError('pinned upstream bootstrap failed; run scripts/bootstrap_upstream.sh or supply FASTRL_GIT_SOURCE=<offline bundle>; tests cannot use missing/unverified source') from e
