#!/usr/bin/env bash
set -euo pipefail

# Standard single-node multi-GPU Diffusion-PPO launch script for ViT-H.
# Configure dataset/tracker paths and optional SWANLAB_API_KEY via environment
# variables or a local git-ignored .env file (never commit secrets).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

: "${VITH_DATASET_ROOT:?Set VITH_DATASET_ROOT to the extracted training_set directory}"
: "${TRACKER_PYTHON_BIN:?Set TRACKER_PYTHON_BIN to the isolated vith-tracker python binary}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/ppo_run_$(date +%Y%m%d_%H%M%S)}"
SOURCES="${PPO_SOURCES:-h36m bedlam bedlam2}"
G1_PTH_NAME="${VITH_G1_PTH_NAME:-g1_new.pth}"
TRACKER_WORKERS="${PPO_TRACKER_WORKERS:-10}"
PPO_SWANLAB_PROJECT="${PPO_SWANLAB_PROJECT:-vit-h-ppo}"
# SwanLab >=0.10 reserves SWANLAB_PROJECT for JSON-encoded nested project settings.
unset SWANLAB_PROJECT

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${VITH_BASE_PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  TORCHRUN_BIN="torchrun"
fi

# shellcheck disable=SC2206
SOURCES_ARR=(${SOURCES})

exec "${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  post_training/train_ppo.py \
  --dataset-root "${VITH_DATASET_ROOT}" \
  --sources "${SOURCES_ARR[@]}" \
  --g1-pth-name "${G1_PTH_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --tracker-python "${TRACKER_PYTHON_BIN}" \
  --tracker-workers "${TRACKER_WORKERS}" \
  --swanlab-project "${PPO_SWANLAB_PROJECT}" \
  "$@"
