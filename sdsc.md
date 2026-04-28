# Submitting jobs on SDSC

SDSC's login nodes are shared and throttled — never run heavy work there. Dispatch GPU / long-CPU work to compute nodes via SLURM.

## Three ways to run

| Use | Command | Notes |
|---|---|---|
| Fire-and-forget (training, sweeps, long eval) | `sbatch job.sh` or `sbatch --wrap="..."` | Job lands in queue, writes to stdout/stderr files, frees your shell |
| Interactive (debugging, inspecting model outputs live) | `srun ... --pty bash -c "..."` | Blocks your shell until the job runs; cleanest for a single short command |
| Persistent interactive shell | `srun ... --pty bash` | Interactive shell on a node; `exit` releases it |

## Standard flags (adjust as needed)

```bash
--partition=nairr-gpu-shared    # GPU-shared partition we have access to
--account=ddp477                # our allocation
--gpus=1                        # GPU count; shared partition supports fractional/single GPUs
--ntasks-per-node=1
--nodes=1
--mem=32G                       # memory; bump for big batches
--time=HH:MM:SS                 # walltime; job is killed at this limit
--job-name=<name>               # shows in squeue; helpful for filtering
--output=logs/<name>_%j.out     # stdout ( %j = JobID )
--error=logs/<name>_%j.err      # stderr (optional; combined if omitted)
```

Rough walltime guidance: short eval (5–30 min), single-model training (1–3 h), LAT sweep (6–24 h), long GCG run (6–12 h). Set it to ~1.5× your best estimate.

## sbatch: background submission

Two equivalent styles. Pick whichever is less fiddly for the task.

### Inline via `--wrap`
```bash
sbatch --job-name=mytask \
       --output=logs/mytask_%j.out \
       --partition=nairr-gpu-shared --account=ddp477 \
       --gpus=1 --ntasks-per-node=1 --nodes=1 \
       --time=02:00:00 --mem=32G \
       --wrap="conda run -n myenv --no-capture-output python script.py --arg foo"
```

### Heredoc (cleaner for multi-line jobs)
```bash
sbatch <<EOF
#!/bin/bash
#SBATCH --partition=nairr-gpu-shared
#SBATCH --account=ddp477
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --job-name=mytask
#SBATCH --output=logs/mytask_%j.out

module load cpu/0.15.4
module load anaconda3/2020.11
eval "\$(conda shell.bash hook)"
conda activate myenv
cd ${WORKDIR}

python script.py --arg foo
EOF
```

## srun: interactive / blocking

Use srun when you want to see output live or step through a short task.

```bash
srun --partition=nairr-gpu-shared \
     --account=ddp477 \
     --gpus=1 --ntasks-per-node=1 --nodes=1 \
     --time=00:30:00 --mem=32G \
     --job-name=quicktest \
     --pty bash -c "conda run -n myenv --no-capture-output python quick_eval.py"
```

`--pty bash` (no trailing command) gives you an interactive shell on the node instead.

## Conda activation inside the job

Because login-node env isn't inherited, the job script must re-activate conda:

```bash
module load cpu/0.15.4
module load anaconda3/2020.11
eval "$(conda shell.bash hook)"
conda activate myenv
```

Or skip `conda activate` entirely and wrap the command: `conda run -n myenv --no-capture-output python ...`. `--no-capture-output` streams stdout live; without it, conda buffers everything.

## Job dependencies

Chain jobs so B only starts after A succeeds:

```bash
JOB_A=$(sbatch --parsable job_a.sh)
sbatch --dependency=afterok:${JOB_A} job_b.sh        # only if A succeeds
sbatch --dependency=afterany:${JOB_A} job_b.sh       # regardless of A's exit
```

Use `afterany` for aggregation jobs that should run even if some upstream jobs fail.

## Monitoring and control

```bash
squeue -u $USER                      # your queued + running jobs
squeue -u $USER --start              # estimated start times
scontrol show job <jobid>            # full details (why pending, resources, etc.)
sacct -u $USER --starttime=today     # finished jobs + exit codes
scancel <jobid>                      # cancel one job
scancel -u $USER                     # cancel all your jobs
```

Watch a log as it grows:
```bash
tail -f logs/mytask_<jobid>.out
```

## Common pitfalls

- **Running on login node**: a Python script that pulls a large model, forward-passes a batch, or trains anything will hammer the shared login node. Always `srun` or `sbatch`. Readonly stuff (small prints, `git pull`, file listings) is fine on login.
- **`cd $WORKDIR` missing**: SLURM submits from your current dir, but a heredoc expands variables at submission, not execution — double-check paths resolve on the compute node.
- **Conda not active inside job**: forgetting to load the conda module or `conda activate`. Jobs fail with `python: command not found` or unexpected interpreter.
- **Walltime too short**: job gets killed mid-training with `CANCELLED DUE TO TIME LIMIT`. No partial recovery; pad walltime.
- **Output buffering**: `python -u` or `conda run --no-capture-output` prevents stdout from getting truncated on kill / showing up minutes late.
- **OOM kills**: shows as `OOM` in `sacct` or a broken pipe in logs. Bump `--mem`.
