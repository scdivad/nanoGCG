#!/bin/bash
#SBATCH --partition=nairr-gpu-shared
#SBATCH --account=ddp477
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --job-name=lg-gcg
#SBATCH --output=logs/lg-gcg_%j.out
set -euo pipefail

# GCG / ACG / I-GCG attack on base Llama Guard 3 8B over a batch of prompts.
#
# Evaluates how easily an adversarial suffix can flip Llama Guard 3's
# classification from "unsafe" to "safe" on each prompt. Uses the batched
# driver in examples/llama_guard.py (one model load amortized across all
# prompts; results appended to JSONL as they're produced).
#
# Per-step times measured on A100-PCIE-40GB: ACG ~0.87 s/step, I-GCG
# ~1.04 s/step. On H100 SXM5 expect ~1.7-1.8x speedup (~0.5-0.6 s/step).
# At num_steps=100, 100 prompts takes ~85-100 min on an H100 before
# --skip-already-safe discounts the safe-by-default prompts. 3h walltime
# leaves a comfortable cushion.
#
# Environment overrides:
#   NANOGCG_REPO      path to this repo (defaults to $SLURM_SUBMIT_DIR)
#   LLAMA_GUARD_PATH  path to Llama-Guard-3-8B weights
#   PROMPTS_FILE      path to one-prompt-per-line text file
#   MODE              gcg | acg | i-gcg  (default: i-gcg)
#   NUM_STEPS         GCG iterations per prompt (default: 100)
#   CONDA_ENV         conda env name (default: gcg)

# Under sbatch, $BASH_SOURCE points at SLURM's spool copy of the script,
# not the real job.sh, so derive REPO from $SLURM_SUBMIT_DIR instead.
REPO="${NANOGCG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"

if [ ! -f "$REPO/examples/llama_guard.py" ]; then
    echo "ERROR: \$REPO=$REPO doesn't contain examples/llama_guard.py" >&2
    echo "       cd to the nanoGCG repo root before 'sbatch job.sh'," >&2
    echo "       or set NANOGCG_REPO=/path/to/nanoGCG." >&2
    exit 1
fi

cd "$REPO"
mkdir -p logs results

MODEL_PATH="${LLAMA_GUARD_PATH:-/home/davidsc2/FOCAL/ctlm/pulled/Llama-Guard-3-8B}"
PROMPTS_FILE="${PROMPTS_FILE:-$REPO/prompts.txt}"
MODE="${MODE:-i-gcg}"
NUM_STEPS="${NUM_STEPS:-100}"
CONDA_ENV="${CONDA_ENV:-gcg}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: \$MODEL_PATH=$MODEL_PATH doesn't exist." >&2
    echo "       Set LLAMA_GUARD_PATH to the Llama Guard weights dir." >&2
    exit 1
fi

if [ ! -f "$PROMPTS_FILE" ]; then
    echo "ERROR: \$PROMPTS_FILE=$PROMPTS_FILE doesn't exist." >&2
    echo "       One prompt per line; blank lines and '#'-comments skipped." >&2
    exit 1
fi

# Reduce CUDA allocator fragmentation under GCG's search_width batch churn.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate conda env (login-node env isn't inherited on compute nodes).
module load cpu/0.15.4
module load anaconda3/2020.11
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

OUT_FILE="results/attack_${SLURM_JOB_ID:-local}_${MODE}.jsonl"
PT_DIR="results/pt_${SLURM_JOB_ID:-local}_${MODE}"

echo "[job] host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "[job] cwd=$PWD  job_id=${SLURM_JOB_ID:-unknown}  conda_env=$CONDA_ENV"
echo "[job] model=$MODEL_PATH"
echo "[job] prompts=$PROMPTS_FILE  mode=$MODE  num_steps=$NUM_STEPS"
echo "[job] results -> $OUT_FILE"
echo "[job] pt dir  -> $PT_DIR"

# -u: unbuffered stdout/stderr so tail -f on the log is live.
VERIFY_EVERY="${VERIFY_EVERY:-20}"

# Disable nanogcg's token-space early_stop (false-positives on Llama 3's BPE
# boundary) and use --verify-every instead: every K steps the driver decodes
# the current best suffix, runs the real Llama Guard classification pipeline,
# and early-stops only when the attack actually flips to "safe" end-to-end.
python -u examples/llama_guard.py \
    --model "$MODEL_PATH" \
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

echo "[job] done. results: $OUT_FILE"
