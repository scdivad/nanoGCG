#!/bin/bash
#SBATCH --array=0-3
#SBATCH --partition=nairr-gpu-shared
#SBATCH --account=ddp477
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --job-name=lg-gcg-sweep
#SBATCH --output=logs/lg-gcg-sweep_%A_%a.out
set -euo pipefail

# I-GCG attack sweep over 4 LAT-hardened Llama Guard checkpoints. Each
# array task takes one GPU, so all 4 run in parallel when the nairr shared
# pool has capacity.
#
# Submit:              sbatch job_array.sh
# Re-run one task:     sbatch --array=2 job_array.sh
# Re-run a subset:     sbatch --array=0,2 job_array.sh
# Status:              squeue -u $USER -j <array_job_id>
# Results layout:      results/sweep_<ARRAY_JOB_ID>_<LABEL>_i-gcg.jsonl
#                      results/sweep_<ARRAY_JOB_ID>_<LABEL>_i-gcg_pt/*.pt
#
# Note: in both training runs ``checkpoint_2`` and ``final`` are the same
# weights, so the sweep only runs on (checkpoint_1, final) per directory.

# --- Model table. Index via $SLURM_ARRAY_TASK_ID. ---
LABELS=(
    "malatang_ckpt1"
    "malatang_final"
    "latguard_ckpt1"
    "latguard_final"
)
ADAPTERS=(
    "/home/dcheung2/new/guard_lat/malatang_out/48428614/checkpoint_1"
    "/home/dcheung2/new/guard_lat/malatang_out/48428614/final"
    "/home/dcheung2/new/guard_lat/lat_guard_out/48428600/checkpoint_1"
    "/home/dcheung2/new/guard_lat/lat_guard_out/48428600/final"
)

LABEL="${LABELS[$SLURM_ARRAY_TASK_ID]}"
ADAPTER_PATH="${ADAPTERS[$SLURM_ARRAY_TASK_ID]}"

# --- Repo / env setup (mirrors job.sh) ---
REPO="${NANOGCG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -f "$REPO/examples/llama_guard.py" ]; then
    echo "ERROR: \$REPO=$REPO doesn't contain examples/llama_guard.py" >&2
    echo "       Submit from the nanoGCG repo root or set NANOGCG_REPO." >&2
    exit 1
fi
cd "$REPO"
mkdir -p logs results

MODEL_PATH="${LLAMA_GUARD_PATH:-/home/dcheung2/new/guard_lat/Llama-Guard-3-8B}"
PROMPTS_FILE="${PROMPTS_FILE:-$REPO/prompts.txt}"
MODE="${MODE:-i-gcg}"
NUM_STEPS="${NUM_STEPS:-250}"
VERIFY_EVERY="${VERIFY_EVERY:-20}"
CONDA_ENV="${CONDA_ENV:-gcg}"

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "ERROR: ADAPTER_PATH=$ADAPTER_PATH doesn't exist." >&2
    exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH=$MODEL_PATH doesn't exist." >&2
    exit 1
fi
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "ERROR: PROMPTS_FILE=$PROMPTS_FILE doesn't exist." >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load cpu/0.15.4
module load anaconda3/2020.11
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

OUT_FILE="results/sweep_${SLURM_ARRAY_JOB_ID}_${LABEL}_${MODE}.jsonl"
PT_DIR="results/sweep_${SLURM_ARRAY_JOB_ID}_${LABEL}_${MODE}_pt"

echo "[sweep] array_task=$SLURM_ARRAY_TASK_ID  label=$LABEL"
echo "[sweep] host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "[sweep] cwd=$PWD  job_id=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  conda_env=$CONDA_ENV"
echo "[sweep] model=$MODEL_PATH"
echo "[sweep] adapter=$ADAPTER_PATH"
echo "[sweep] prompts=$PROMPTS_FILE  mode=$MODE  num_steps=$NUM_STEPS  verify_every=$VERIFY_EVERY"
echo "[sweep] results -> $OUT_FILE"
echo "[sweep] pt dir  -> $PT_DIR"

# I-GCG attack. No --init-from-jsonl (user requested plain I-GCG, no
# easy-to-hard warm-start). --no-early-stop + --verify-every gives
# string-space success checks that dodge Llama 3's BPE-boundary
# false-positive trap.
python -u examples/llama_guard.py \
    --model "$MODEL_PATH" \
    --adapter-path "$ADAPTER_PATH" \
    --mode "$MODE" \
    --prompts-file "$PROMPTS_FILE" \
    --output-file "$OUT_FILE" \
    --pt-output-dir "$PT_DIR" \
    --num-steps "$NUM_STEPS" \
    --no-early-stop \
    --verify-every "$VERIFY_EVERY" \
    --skip-already-safe \
    --seed 0 \
    --verbosity WARNING

echo "[sweep] done: $LABEL -> $OUT_FILE"
