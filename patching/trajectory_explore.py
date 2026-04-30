"""
Trajectory margin-distribution exploration on Llama Guard 3.

For each successful-attack .pt file in --pt-dir:
  Load prompt_ids (clean), attacked_prompt_ids (final), all_suffix_ids
  (per-GCG-step suffix tokens, length-T list). Locate the suffix region
  via LCP/LCT alignment. For each step t in 0..T-1, rebuild inter_ids[t]
  by replacing the suffix region of attacked_prompt_ids with
  all_suffix_ids[t], append '\\n\\n', forward through Llama Guard, and
  read the verdict logit_diff = safe - unsafe at the last position.

Outputs:
  --out-csv : per-(prompt, step) row with margin + gap_fraction +
              consecutive-step delta. Use this to choose a margin-shift
              threshold tau and a consecutive-step de-dup delta for the
              trajectory-aware attribution patching script.

Usage:
  python patching/trajectory_explore.py \\
      --pt-dir results/pt_48628592_base_acg \\
      --out-csv patching/trajectory_margins_48628592.csv \\
      [--limit 20] [--model-path ...]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_sweep import align_suffix_positions


def to_long_1d(x):
    """Coerce list/tuple/0-d/1-d/2-d tensor to a 1-D LongTensor on CPU."""
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
    else:
        t = torch.as_tensor(x)
    t = t.long().squeeze()
    if t.dim() == 0:
        t = t.unsqueeze(0)
    if t.dim() != 1:
        raise ValueError(f"expected 1-D suffix ids, got shape {tuple(t.shape)}")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of prompts (after success filter)")
    ap.add_argument("--include-final-as-step", action="store_true",
                    help="Also evaluate attacked_prompt_ids itself as a "
                    "step (labelled 'final'); useful for sanity-checking "
                    "that all_suffix_ids[-1] reproduces the saved final.")
    ap.add_argument("--no-verdict-prefix", action="store_true",
                    help="Skip the '\\n\\n' append before reading verdict "
                    "logits. Use for Llama Guard 1 (verdict emitted "
                    "directly after [/INST]).")
    args = ap.parse_args()

    pt_dir = Path(args.pt_dir)
    pt_files = sorted(pt_dir.glob("prompt_*.pt"))
    print(f"Found {len(pt_files)} .pt files in {pt_dir}", flush=True)

    records = []
    for p in pt_files:
        d = torch.load(p, weights_only=False)
        if d.get("post_verdict") == "safe" and d.get("success"):
            if "all_suffix_ids" not in d or "prompt_ids" not in d \
               or "attacked_prompt_ids" not in d:
                print(f"  SKIP {p.name}  missing trajectory keys "
                      f"(have: {sorted(d.keys())})")
                continue
            records.append((p, d))
    print(f"Using {len(records)} successful attacks with trajectory data", flush=True)
    if args.limit:
        records = records[: args.limit]
        print(f"Capped to {len(records)} (--limit)")
    if not records:
        raise SystemExit("Nothing to do.")

    tok_path = args.model_path or records[0][1]["tokenizer_name_or_path"]
    print(f"Loading model: {tok_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(tok_path)
    model = AutoModelForCausalLM.from_pretrained(
        tok_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    dev = model.device
    print("Model ready.", flush=True)

    NEWLINE_TOK = tok("\n\n",   add_special_tokens=False).input_ids[0]
    SAFE_TOK    = tok("safe",   add_special_tokens=False).input_ids[0]
    UNSAFE_TOK  = tok("unsafe", add_special_tokens=False).input_ids[0]
    FILLER_TOK  = tok("!",      add_special_tokens=False).input_ids[0]

    @torch.no_grad()
    def fwd_diff(ids_1d):
        """ids_1d: 1-D LongTensor (no batch). For LG3 append '\\n\\n' to
        set the verdict-predicting position; for LG1 use the ids as-is.
        Return logit(safe) - logit(unsafe) at the final position."""
        if args.no_verdict_prefix:
            ids = ids_1d.unsqueeze(0).to(dev)
        else:
            ids = torch.cat([ids_1d, torch.tensor([NEWLINE_TOK])]).unsqueeze(0).to(dev)
        out = model(input_ids=ids).logits
        return float(out[0, -1, SAFE_TOK] - out[0, -1, UNSAFE_TOK])

    out_rows = []
    t0 = time.time()
    for i, (pt_path, d) in enumerate(records, start=1):
        prompt_ids   = to_long_1d(d["prompt_ids"])
        attacked_ids = to_long_1d(d["attacked_prompt_ids"])
        all_suff     = d["all_suffix_ids"]
        if not isinstance(all_suff, (list, tuple)):
            all_suff = [all_suff]
        all_suff = [to_long_1d(s) for s in all_suff]

        s_start, s_end = align_suffix_positions(prompt_ids, attacked_ids)
        suffix_len = s_end - s_start

        # Length-matched benign baseline (final attacked with '!' at suffix)
        benign_ids = attacked_ids.clone()
        benign_ids[s_start:s_end] = FILLER_TOK
        clean_diff = fwd_diff(benign_ids)
        corr_diff  = fwd_diff(attacked_ids)
        gap        = corr_diff - clean_diff

        # Per-step intermediate diffs
        step_diffs = []
        bad_lens = 0
        for t, suff_t in enumerate(all_suff):
            if suff_t.numel() != suffix_len:
                bad_lens += 1
                step_diffs.append(None)
                continue
            inter = attacked_ids.clone()
            inter[s_start:s_end] = suff_t
            step_diffs.append(fwd_diff(inter))

        # Optional sanity-check: forward attacked_ids as 'final' too
        final_diff = corr_diff if args.include_final_as_step else None

        # Emit rows
        prev = None
        for t, sd in enumerate(step_diffs):
            if sd is None:
                continue
            gap_frac = (sd - clean_diff) / gap if abs(gap) > 1e-9 else float("nan")
            delta_prev = (sd - prev) if prev is not None else None
            out_rows.append({
                "source_file":  pt_path.name,
                "prompt_idx":   i - 1,
                "step":         t,
                "n_steps":      len(all_suff),
                "suffix_len":   suffix_len,
                "clean_diff":   clean_diff,
                "corr_diff":    corr_diff,
                "gap":          gap,
                "inter_diff":   sd,
                "gap_fraction": gap_frac,
                "delta_prev":   delta_prev,
            })
            prev = sd

        elapsed = time.time() - t0
        rate = i / elapsed
        eta = (len(records) - i) / rate if rate > 0 else float("inf")
        print(f"[{i}/{len(records)}] {pt_path.name}  "
              f"clean={clean_diff:+.2f}  corr={corr_diff:+.2f}  gap={gap:+.2f}  "
              f"steps={len(all_suff)} bad_lens={bad_lens}  "
              f"elapsed={elapsed:.0f}s  eta={eta:.0f}s",
              flush=True)

    # Write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_file", "prompt_idx", "step", "n_steps", "suffix_len",
                  "clean_diff", "corr_diff", "gap", "inter_diff",
                  "gap_fraction", "delta_prev"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {len(out_rows)} rows -> {out_csv}", flush=True)

    # Summary stats
    gf = np.array([r["gap_fraction"] for r in out_rows
                   if r["gap_fraction"] == r["gap_fraction"]])
    dp = np.array([r["delta_prev"] for r in out_rows
                   if r["delta_prev"] is not None])
    print("\n=== gap_fraction = (inter_diff - clean_diff) / gap  (1.0 = fully attacked, 0.0 = clean) ===")
    if gf.size:
        for q in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
            print(f"  q{int(q*100):>2}: {np.quantile(gf, q):+.3f}")
        print(f"  mean: {gf.mean():+.3f}  std: {gf.std():+.3f}  n={gf.size}")
    print("\n=== |delta_prev| = |inter_diff[t] - inter_diff[t-1]| (consecutive-step margin shift) ===")
    if dp.size:
        adp = np.abs(dp)
        for q in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
            print(f"  q{int(q*100):>2}: {np.quantile(adp, q):.3f}")
        print(f"  mean: {adp.mean():.3f}  median: {np.median(adp):.3f}  n={adp.size}")


if __name__ == "__main__":
    main()
