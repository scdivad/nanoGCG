#!/bin/bash
#SBATCH --partition=nairr-gpu-shared
#SBATCH --account=ddp477
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --job-name=patch-sweep
#SBATCH --output=logs/patch-sweep_%j.out
set -euo pipefail

# Run the 5-way activation-patching sweep (Q/K/V/O/MLP × layers ×
# {scaffold_header, user_content, adversarial, scaffold_tail, last}) on a
# set of successful GCG attacks. Mirrors examples/llama_guard.py's PEFT
# loading so the hooks see the exact same forward as attack generation.
#
# Env overrides:
#   NANOGCG_REPO      repo root (defaults to $SLURM_SUBMIT_DIR)
#   LLAMA_GUARD_PATH  base Llama Guard 3 8B weights path
#   PT_DIR            directory of .pt files from a prior attack run
#                     (e.g. results/pt_48374202_lat_i-gcg)
#   ADAPTER_PATH      optional PEFT/LoRA adapter to attach on top of the base
#                     (must match the adapter the attacks were generated
#                     against; otherwise hooks see a different network).
#   OUT_DIR           where to write .png + .json
#   CONDA_ENV         conda env (default gcg)
#   PATCH_MODE        'clean' (default) or 'zero'
#
# Typical usage for re-analyzing LAT attacks on SDSC:
#   ADAPTER_PATH=/path/to/lat/adapter \
#   PT_DIR=results/pt_48374202_lat_i-gcg \
#   OUT_DIR=patching/lat_suffix_5way_fixed \
#   sbatch patching/patch_job.sh

REPO="${NANOGCG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"

if [ ! -f "$REPO/patching/patch_sweep.py" ]; then
    echo "ERROR: \$REPO=$REPO doesn't contain patching/patch_sweep.py" >&2
    echo "       Run 'sbatch patching/patch_job.sh' from the repo root," >&2
    echo "       or set NANOGCG_REPO=/path/to/nanoGCG." >&2
    exit 1
fi

cd "$REPO"
mkdir -p logs

MODEL_PATH="${LLAMA_GUARD_PATH:-/home/dcheung2/new/Llama-Guard-3-8B}"
PT_DIR_ARG="${PT_DIR:-}"
OUT_DIR_ARG="${OUT_DIR:-}"
CONDA_ENV="${CONDA_ENV:-gcg}"
PATCH_MODE="${PATCH_MODE:-clean}"
ADAPTER_PATH="${ADAPTER_PATH:-}"

if [ -z "$PT_DIR_ARG" ]; then
    echo "ERROR: set PT_DIR to a directory with prompt_*.pt files." >&2
    exit 1
fi
if [ ! -d "$PT_DIR_ARG" ]; then
    echo "ERROR: PT_DIR=$PT_DIR_ARG doesn't exist." >&2
    exit 1
fi
if [ -z "$OUT_DIR_ARG" ]; then
    # Default: infer a name from the pt dir
    base=$(basename "$PT_DIR_ARG")
    OUT_DIR_ARG="patching/sweep_${base}_${PATCH_MODE}"
fi

ADAPTER_FLAG=()
TARGET_TAG="base"
if [ -n "$ADAPTER_PATH" ]; then
    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "ERROR: ADAPTER_PATH=$ADAPTER_PATH doesn't exist." >&2
        exit 1
    fi
    ADAPTER_FLAG=(--adapter-path "$ADAPTER_PATH")
    TARGET_TAG="lat"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module load cpu/0.15.4
module load anaconda3/2020.11
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

echo "[job] host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "[job] cwd=$PWD  job_id=${SLURM_JOB_ID:-unknown}  target=${TARGET_TAG}"
echo "[job] model=$MODEL_PATH"
echo "[job] pt-dir=$PT_DIR_ARG  out-dir=$OUT_DIR_ARG  patch-mode=$PATCH_MODE"
[ -n "$ADAPTER_PATH" ] && echo "[job] adapter=$ADAPTER_PATH"

python -u patching/patch_sweep.py \
    --pt-dir "$PT_DIR_ARG" \
    --out-dir "$OUT_DIR_ARG" \
    --model-path "$MODEL_PATH" \
    --patch-mode "$PATCH_MODE" \
    "${ADAPTER_FLAG[@]}"

echo "[job] done. outputs in $OUT_DIR_ARG"
