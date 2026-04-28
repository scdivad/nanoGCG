# Quickstart — minimal files to get oriented

This guide tells a new collaborator the **smallest set of files** they
need to read to:

1. Find and load cached adversarial suffixes from a completed ACG / I-GCG run.
2. Load Llama Guard 3 8B and run an attacked prompt through it.
3. Install activation hooks on MLP neurons for interp / patching work.

If you just want to *run* a new attack, see [README.md](README.md) and
[job.sh](job.sh). This doc is for working with already-produced results.

---

## 1. Cached attack results — file layout

> **Where the data lives.** The large attack runs (e.g. the 100-prompt
> ACG sweep `attack_48411899_base_acg.jsonl` — 90% ASR — and the LAT
> sweep arrays) live on **SDSC** at `/home/dcheung2/new/nanoGCG/results/`.
> Only a few small dev runs are checked into the local clone. To work
> with the SDSC results either `scp` them down or `srun` an interactive
> session on the compute node.

Each completed attack run produces two artifacts:

```
results/attack_<jobid>_<base|lat>_<mode>.jsonl   # one JSON line per prompt
results/pt_<jobid>_<base|lat>_<mode>/            # one .pt per prompt
    prompt_001.pt
    prompt_002.pt
    ...
```

The JSONL is a quick textual summary — open it with `head` or `jq` to
scan ASR. The `.pt` files carry the actual tokenized state needed to
replay an attack through the model (no re-tokenization round-trip).

### JSONL record (one line per prompt)

Read just **one line** of any `attack_*.jsonl` to learn the schema:

```python
import json
row = json.loads(open("results/attack_48411899_base_acg.jsonl").readline())
# row.keys() = {
#   'prompt', 'attacked_prompt', 'best_suffix',
#   'pre_classification', 'pre_verdict', 'pre_category',
#   'post_classification', 'post_verdict', 'post_category',
#   'best_loss', 'num_steps_run', 'success', 'elapsed_sec',
#   'adv_position',        # 'suffix' | 'prefix'
# }
```

ASR sweep:

```bash
for f in results/attack_*.jsonl; do
    total=$(wc -l < "$f")
    succ=$(grep -c '"success": true' "$f")
    echo "$f  $succ/$total"
done
```

### .pt record (one file per prompt)

Read **one** `.pt` from any `pt_*` directory — that's the canonical
schema:

```python
import torch
d = torch.load("results/pt_local_i-gcg/prompt_001.pt", weights_only=False)
# d.keys() = [
#   'prompt', 'prompt_ids',                # original chat-template-wrapped IDs
#   'attacked_prompt', 'attacked_prompt_ids',  # with suffix/prefix injected
#   'best_suffix', 'suffix_ids',           # the adversarial tokens (typically 20)
#   'best_loss', 'num_steps_run', 'success',
#   'pre_verdict', 'pre_category',
#   'post_verdict', 'post_category',
#   'target',                              # '\n\nsafe'
#   'mode',                                # 'gcg' | 'acg' | 'i-gcg'
#   'tokenizer_name_or_path',
# ]
```

**Important:** use `attacked_prompt_ids` (and not the decoded
`attacked_prompt` string) when you want to feed the attack back into
the model. Llama 3's BPE doesn't always round-trip cleanly, and
re-tokenizing the string can produce different IDs that no longer
trigger the flip. The whole reason the IDs are cached is to skip that
round-trip.

---

## 2. Load Llama Guard and run an attacked prompt

The minimum viable script:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/davidsc2/FOCAL/ctlm/pulled/Llama-Guard-3-8B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda:0",
).eval()

# Replay a cached attack
d = torch.load("results/pt_local_i-gcg/prompt_001.pt", weights_only=False)
ids = d["attacked_prompt_ids"].unsqueeze(0).to(model.device)

# Method A: real classification (greedy generation)
out = model.generate(ids, max_new_tokens=10, do_sample=False,
                     pad_token_id=tok.eos_token_id)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
# -> "\n\nsafe" if attack worked, else "\n\nunsafe\nS<n>"

# Method B: score-based (no generation, faster)
# Append "\n\n" because Llama Guard's first generated token is always
# "\n\n" — safe vs unsafe diverge at the *next* position.
nn = torch.tensor(tok.encode("\n\n", add_special_tokens=False),
                  device=model.device).unsqueeze(0)
ids_scored = torch.cat([ids, nn], dim=1)
with torch.no_grad():
    logits = model(input_ids=ids_scored, use_cache=False).logits[0, -1, :]
safe_id   = tok.encode("safe",   add_special_tokens=False)[0]   # 19193
unsafe_id = tok.encode("unsafe", add_special_tokens=False)[0]   # 39257
margin = (logits[unsafe_id] - logits[safe_id]).item()
print(f"unsafe-margin = {margin:+.2f}  ->  {'unsafe' if margin > 0 else 'safe'}")
```

**For LAT-adapter checkpoints** (LoRA-finetuned Llama Guard), wrap with PEFT:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(model, "/path/to/adapter_dir")
```

Production references — read these once if you need a feature already
implemented:

- [examples/llama_guard.py](examples/llama_guard.py) — full attack driver
  (chat template handling, batched prompt loop, `--adapter-path`,
  `--verify-every`, JSONL output).
- [examples/clean_accuracy.py](examples/clean_accuracy.py) — clean-vs-harmful
  recall probe (the script that caught defensive collapse on a LAT checkpoint).
- [harm_classifier_benchmark/run_benchmark.py](harm_classifier_benchmark/run_benchmark.py) —
  reusable `LlamaGuardBackend.classify()` that handles the `\n\n` scoring trick.

---

## 3. Hooks on Llama Guard MLP neurons

Llama's per-layer MLP is:

```
down_proj( SiLU(gate_proj(x)) * up_proj(x) )
                                 └── shape (..., 14336) — the gated neurons
```

The argument to `down_proj` is the post-gating, pre-projection tensor —
the natural per-neuron interpretation. Register a `forward_pre_hook` on
`down_proj` to capture or modify it.

### Capture activations

```python
captured = {}
def make_hook(layer_idx):
    def hook(module, args):
        # args is a 1-tuple: (input_tensor,)
        x = args[0]                      # shape (B, T, 14336)
        captured[layer_idx] = x.detach().clone()
    return hook

handles = []
for L in [0, 5, 10, 15]:
    h = model.model.layers[L].mlp.down_proj.register_forward_pre_hook(make_hook(L))
    handles.append(h)

# Forward pass (or generate, etc.) populates `captured`
with torch.no_grad():
    model(input_ids=ids)

for h in handles:
    h.remove()
```

### Patch (overwrite) specific neurons at specific positions

```python
def make_patch_hook(layer_idx, replacement, positions, neurons):
    """positions: list[int] (token positions to patch)
       neurons:   list[int] (neuron indices to overwrite)
       replacement: tensor of shape (len(positions), len(neurons))"""
    def hook(module, args):
        x = args[0].clone()              # (B, T, 14336)
        for i, p in enumerate(positions):
            x[:, p, neurons] = replacement[i]
        return (x,)                      # return modified args tuple
    return hook
```

The hook **must** return the modified `args` tuple (not the bare
tensor) — `register_forward_pre_hook` expects positional-arg
replacement.

Worked example: [patching/mlp_sparse_patch.py](patching/mlp_sparse_patch.py)
— sparse top-k MLP-neuron patching across layers, with both
no-grad capture and grad-enabled patching variants. The two helper
functions to read are `make_capture_nograd` and the patching block
around line 325.

For attention-pattern hooks, see
[patching/attn_pattern_patch.py](patching/attn_pattern_patch.py)
(same pattern, different module).

---

## 4. Running new attacks on SDSC

For SLURM basics (partition flags, walltime guidance, srun vs sbatch
patterns), see [sdsc.md](sdsc.md). Below are the nanoGCG-specific entry
points.

### Single-target attack (`job.sh`)

```bash
# Base Llama Guard, AdvBench-style prompts (default)
sbatch job.sh

# Pick the mode and prompts file
MODE=i-gcg PROMPTS_FILE=/path/to/aegis_prompts.jsonl sbatch job.sh

# Attack a LAT-adapter checkpoint
ADAPTER_PATH=/home/dcheung2/new/guard_lat/lat_guard_out/<jobid>/checkpoint_1 \
LLAMA_GUARD_PATH=/home/dcheung2/new/guard_lat/Llama-Guard-3-8B \
PROMPTS_FILE=/home/dcheung2/new/nanoGCG/aegis_prompts.jsonl \
    sbatch job.sh

# Pilot run on first N prompts
LIMIT=10 sbatch job.sh
```

Env-var overrides supported by [job.sh](job.sh): `MODE`, `NUM_STEPS`,
`PROMPTS_FILE`, `LLAMA_GUARD_PATH`, `ADAPTER_PATH`, `LIMIT`,
`VERIFY_EVERY`, `RESUME_FROM`, `INIT_FROM_JSONL`, `CONDA_ENV`.

Output lands in `results/attack_<slurm_jobid>_<base|lat>_<mode>.jsonl`
plus a `results/pt_<slurm_jobid>_<base|lat>_<mode>/` directory of `.pt`
files (one per prompt). The `<base|lat>` tag depends on whether
`ADAPTER_PATH` was set.

### Parallel sweep over multiple checkpoints (`job_array.sh`)

`job_array.sh` runs a hard-coded list of (label, adapter-path) pairs as
a SLURM job array — one task per GPU.

```bash
sbatch job_array.sh                       # all tasks (--array=0-3)
sbatch --array=2 job_array.sh             # one task
sbatch --array=0,2 job_array.sh           # subset
```

Edit the `LABELS` and `ADAPTERS` arrays inside the script to add a new
checkpoint. Output filename is
`results/sweep_<arrayjobid>_<label>_<mode>.jsonl`.

### Clean-accuracy probe (`job_clean_acc.sh`)

Sanity-check that a checkpoint hasn't collapsed (predicting the same
class for everything). Reports safe-recall + unsafe-recall on the Aegis
test split.

```bash
sbatch job_clean_acc.sh                                            # base model
ADAPTER_PATH=/path/to/adapter sbatch job_clean_acc.sh              # LAT adapter
N_PER_CLASS=30 ADAPTER_PATH=/path/to/adapter sbatch job_clean_acc.sh
```

Output: `logs/lg-clean-acc_<jobid>.out`. Always run this on a new LAT
checkpoint **before** spending GPU-hours on an attack — see
[examples/clean_accuracy.py](examples/clean_accuracy.py) for the script
that's invoked. We discovered `lat_guard_out/48452844/checkpoint_1`
predicts "unsafe" for 100/100 safe-labeled prompts (full collapse), so
its 0% ASR result is meaningless.

### Interactive GPU session (`srun`)

Use this when you want to inspect a cached `.pt` interactively, debug an
attack on a single prompt, or run the clean-accuracy snippet by hand
without writing an sbatch wrapper. Login nodes have no GPU — anything
that loads Llama Guard must go through `srun` or `sbatch`. See
[sdsc.md](sdsc.md) for the full SLURM reference.

```bash
# Grab one A100 for an hour. Wrap in tmux first so an ssh drop
# doesn't kill the allocation.
tmux new -s gcg            # detach: Ctrl-b d ; reattach: tmux a -t gcg

srun --partition=nairr-gpu-shared --account=ddp477 \
     --gpus=1 --ntasks-per-node=1 --nodes=1 \
     --time=01:00:00 --mem=48G \
     --pty bash

# --- now on the compute node ---
module load cpu/0.15.4
module load anaconda3/2020.11
eval "$(conda shell.bash hook)"
conda activate gcg
cd /home/dcheung2/new/nanoGCG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Replay a cached attack against the base model
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained('/home/dcheung2/new/guard_lat/Llama-Guard-3-8B')
model = AutoModelForCausalLM.from_pretrained(
    '/home/dcheung2/new/guard_lat/Llama-Guard-3-8B',
    dtype=torch.bfloat16, device_map='cuda:0').eval()
d = torch.load('results/pt_48411899_base_acg/prompt_001.pt', weights_only=False)
ids = d['attacked_prompt_ids'].unsqueeze(0).to('cuda:0')
out = model.generate(ids, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
print('verdict:', tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
"
```

**One-shot variant** — runs a command on a compute node and exits, no
shell:

```bash
srun --partition=nairr-gpu-shared --account=ddp477 \
     --gpus=1 --ntasks-per-node=1 --nodes=1 \
     --time=00:15:00 --mem=48G \
     --pty bash -c "cd /home/dcheung2/new/nanoGCG && \
                    conda run -n gcg --no-capture-output \
                    python examples/clean_accuracy.py --n-per-class 30"
```

Output streams live to your terminal. Good for short evals (<15min);
for anything longer use `sbatch` so it survives an ssh drop.

### Monitoring jobs

```bash
squeue -u $USER                       # what's queued/running
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS  # post-mortem
tail -f logs/lg-gcg_<jobid>.out       # live log

# Rolling ASR while the job is still running
wc -l results/attack_<jobid>_*.jsonl
grep -c '"success": true' results/attack_<jobid>_*.jsonl
```

---

## TL;DR — the four files to read

| Purpose | File | Lines worth reading |
|---|---|---|
| Cached results format | any `pt_*/prompt_001.pt` | (load and `print(d.keys())`) |
| Run an attacked prompt | this doc, section 2 | — |
| Reference attack pipeline | [examples/llama_guard.py](examples/llama_guard.py) | `classify()`, `attack_one()` |
| MLP hook recipe | [patching/mlp_sparse_patch.py](patching/mlp_sparse_patch.py) | `make_capture_nograd`, hook around line 325 |
