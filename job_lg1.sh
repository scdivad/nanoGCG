#!/bin/bash
#SBATCH --partition=nairr-gpu-shared
#SBATCH --account=ddp477
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --job-name=lg1-gcg
#SBATCH --output=logs/lg1-gcg_%j.out
set -euo pipefail

# GCG / ACG / I-GCG attack on Llama Guard 1 (7B) over a batch of prompts.
# Sister script to job.sh (which targets Llama Guard 3 8B). Calls the
# llama_guard_1.py driver — same JSONL + .pt trajectory output schema.
#
# LG1 is smaller (7B vs 8B) and uses the Llama 2 chat format, so per-step
# wall time is somewhat lower than LG3 but the search budget needs are
# similar.
#
# Environment overrides (same set as job.sh):
#   NANOGCG_REPO      path to this repo (defaults to $SLURM_SUBMIT_DIR)
#   LLAMA_GUARD_PATH  path to LlamaGuard-7b weights
#   PROMPTS_FILE      path to prompts (.txt one-per-line or .jsonl)
#   MODE              gcg | acg | i-gcg  (default: i-gcg)
#   NUM_STEPS         GCG iterations per prompt (default: 250)
#   ADAPTER_PATH      optional PEFT/LoRA adapter dir
#   RESUME_FROM       JSONL of prior run; skip its prompts
#   INIT_FROM_JSONL   warm-start from prior run's best suffix
#   LIMIT             cap prompts attacked
#   CONDA_ENV         conda env name (default: gcg)

REPO="${NANOGCG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"

if [ ! -f "$REPO/examples/llama_guard_1.py" ]; then
    echo "ERROR: \$REPO=$REPO doesn't contain examples/llama_guard_1.py" >&2
    echo "       cd to the nanoGCG repo root before 'sbatch job_lg1.sh'," >&2
    echo "       or set NANOGCG_REPO=/path/to/nanoGCG." >&2
    exit 1
fi

cd "$REPO"
mkdir -p logs results

MODEL_PATH="${LLAMA_GUARD_PATH:-/home/dcheung2/new/guard_lat/LlamaGuard-7b}"
PROMPTS_FILE="${PROMPTS_FILE:-$REPO/prompts.jsonl}"
MODE="${MODE:-i-gcg}"
NUM_STEPS="${NUM_STEPS:-250}"
CONDA_ENV="${CONDA_ENV:-gcg}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: \$MODEL_PATH=$MODEL_PATH doesn't exist." >&2
    echo "       Set LLAMA_GUARD_PATH to the LlamaGuard-7b weights dir." >&2
    exit 1
fi

if [ ! -f "$PROMPTS_FILE" ]; then
    echo "ERROR: \$PROMPTS_FILE=$PROMPTS_FILE doesn't exist." >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load cpu/0.15.4
module load anaconda3/2020.11
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

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

OUT_FILE="results/attack_${SLURM_JOB_ID:-local}_lg1_${TARGET_TAG}_${MODE}.jsonl"
PT_DIR="results/pt_${SLURM_JOB_ID:-local}_lg1_${TARGET_TAG}_${MODE}"

echo "[job] host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "[job] cwd=$PWD  job_id=${SLURM_JOB_ID:-unknown}  conda_env=$CONDA_ENV"
echo "[job] model=$MODEL_PATH"
echo "[job] adapter=${ADAPTER_PATH:-<none - attacking base model>}"
echo "[job] prompts=$PROMPTS_FILE  mode=$MODE  num_steps=$NUM_STEPS"
echo "[job] results -> $OUT_FILE"
echo "[job] pt dir  -> $PT_DIR"

VERIFY_EVERY="${VERIFY_EVERY:-20}"

RESUME_FROM="${RESUME_FROM:-}"
RESUME_FLAG=()
if [ -n "$RESUME_FROM" ]; then
    if [ ! -f "$RESUME_FROM" ]; then
        echo "ERROR: RESUME_FROM=$RESUME_FROM doesn't exist." >&2
        exit 1
    fi
    RESUME_FLAG=(--resume-from "$RESUME_FROM")
    echo "[job] resume-from -> $RESUME_FROM"
fi

INIT_FROM_JSONL="${INIT_FROM_JSONL:-}"
INIT_FROM_FLAG=()
if [ -n "$INIT_FROM_JSONL" ]; then
    if [ ! -f "$INIT_FROM_JSONL" ]; then
        echo "ERROR: INIT_FROM_JSONL=$INIT_FROM_JSONL doesn't exist." >&2
        exit 1
    fi
    INIT_FROM_FLAG=(--init-from-jsonl "$INIT_FROM_JSONL")
    echo "[job] init-from-jsonl -> $INIT_FROM_JSONL"
fi

LIMIT="${LIMIT:-}"
LIMIT_FLAG=()
if [ -n "$LIMIT" ]; then
    LIMIT_FLAG=(--limit "$LIMIT")
    echo "[job] limit -> $LIMIT prompts"
fi

python -u examples/llama_guard_1.py \
    --model "$MODEL_PATH" \
    "${ADAPTER_FLAG[@]}" \
    --mode "$MODE" \
    --prompts-file "$PROMPTS_FILE" \
    --output-file "$OUT_FILE" \
    --pt-output-dir "$PT_DIR" \
    --num-steps "$NUM_STEPS" \
    --no-early-stop \
    --verify-every "$VERIFY_EVERY" \
    --skip-already-safe \
    "${RESUME_FLAG[@]}" \
    "${INIT_FROM_FLAG[@]}" \
    "${LIMIT_FLAG[@]}" \
    --seed 0 \
    --verbosity WARNING

echo "[job] done. results: $OUT_FILE"
