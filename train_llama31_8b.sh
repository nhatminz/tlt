#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_KEY="llama31_8b" METHOD="tlt_opd_reflex"
# All hyperparameters are shared in run_rl.sh / scripts/launch_common.sh.
# Override CUDA_VISIBLE_DEVICES, MODEL, dataset and upstream Hydra fields via env/args.
bash "$ROOT/scripts/run_tlt_opd_reflex.sh" "$@"
