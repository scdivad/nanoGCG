#!/bin/bash
#SBATCH --partition=nairr-gpu-shared
#SBATCH --account=ddp477
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --mem=48G
#SBATCH --job-name=lg-clean-acc
#SBATCH --output=logs/lg-clean-acc_%j.out
set -euo pipefail

# Clean-accuracy probe for a Llama Guard checkpoint (base or LAT-adapter).
# Reports safe-recall + unsafe-recall on the Aegis test split.
#
# Submit with overrides, e.g.:
#   ADAPTER_PATH=/home/dcheung2/new/guard_lat/lat_guard_out/48452844/checkpoint_1 \
#       sbatch job_clean_acc.sh
#
#   N_PER_CLASS=30 sbatch job_clean_acc.sh             # smaller sample
#   sbatch job_clean_acc.sh                            # base model, default 100/class
#
# Output goes to logs/lg-clean-acc_<jobid>.out

REPO="${NANOGCG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -f "$REPO/examples/clean_accuracy.py" ]; then
    echo "ERROR: \$REPO=$REPO doesn't contain examples/clean_accuracy.py" >&2
    exit 1
fi
cd "$REPO"
mkdir -p logs

MODEL_PATH="${LLAMA_GUARD_PATH:-/home/dcheung2/new/guard_lat/Llama-Guard-3-8B}"
DATASET_PATH="${DATASET_PATH:-/home/dcheung2/new/harm_classifiers/datasets/aegis/test}"
N_PER_CLASS="${N_PER_CLASS:-100}"
CONDA_ENV="${CONDA_ENV:-gcg}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH=$MODEL_PATH doesn't exist." >&2
    exit 1
fi
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: DATASET_PATH=$DATASET_PATH doesn't exist." >&2
    exit 1
fi

ADAPTER_PATH="${ADAPTER_PATH:-}"
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

echo "[clean-acc] host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "[clean-acc] cwd=$PWD  job_id=${SLURM_JOB_ID:-unknown}  conda_env=$CONDA_ENV"
echo "[clean-acc] model=$MODEL_PATH"
echo "[clean-acc] adapter=${ADAPTER_PATH:-<none - probing base model>}"
echo "[clean-acc] dataset=$DATASET_PATH  n_per_class=$N_PER_CLASS  target=$TARGET_TAG"

python -u examples/clean_accuracy.py \
    --model "$MODEL_PATH" \
    "${ADAPTER_FLAG[@]}" \
    --dataset-path "$DATASET_PATH" \
    --n-per-class "$N_PER_CLASS"

echo "[clean-acc] done."
