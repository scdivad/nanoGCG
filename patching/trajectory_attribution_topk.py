"""
Trajectory-augmented attribution patching on Llama Guard 3.

Standard attribution patching (e.g. mlp_sparse_patch.py) uses ONE state
per attack: the final converged adversarial suffix. This script extends
that by taking the GCG optimisation trajectory (all_suffix_ids stored
per step) and accumulating attribution scores across MANY (prompt, step)
pairs. Hypothesis: more diverse attribution data sharpens the per-layer
top-K recovery curve, so fewer neurons are needed for the same recovery.

Per layer L in --layers (default 0..12 inclusive), for each surviving
(prompt, step) entry:

  inter_ids[t]  := attacked_prompt_ids with the suffix region replaced by
                   all_suffix_ids[t] (length-matched).
  attribution[L, n] += sum_{p in suffix} (clean_act[L, p, n] - inter_act[L, p, n])
                                         * d(margin) / d(inter_act[L, p, n])

Per-prompt attribution is averaged across that prompt's surviving steps,
then averaged across prompts (matches the per-prompt-normalised
weighting from cluster_neurons_attribution_topk.py).

Filters applied to a step before it counts:
  1. suffix-identity dedup: drop steps whose all_suffix_ids[t] equals
     all_suffix_ids[t-1] (~50% of trajectory entries per the
     trajectory_explore.py distribution; pure GCG bookkeeping waste).
  2. margin-shift threshold: drop steps with
       gap_fraction(t) = (inter_diff[t] - clean_diff) / gap  <  tau
     The script accumulates four parallel per-layer attribution buffers
     for tau in {0.0, 0.25, 0.5, "final"} so all four ranking + recovery
     curves come out of a single forward+backward sweep.

Recovery is always measured on the final attacked state (corr): patch
top-K neurons of layer L at the suffix positions to their length-matched
'!'-filler clean values, read off (corr_diff - patched_diff) / gap.
This mirrors mlp_sparse_patch.py — only the *neuron ranking* uses
trajectory data; the recovery measurement is unchanged for fair
comparison with the existing topk plot.

Usage:
  python patching/trajectory_attribution_topk.py \\
      --pt-dir results/pt_48628592_base_acg \\
      --out-dir patching/trajectory_attribution \\
      --layers 0,1,2,3,4,5,6,7,8,9,10,11,12 \\
      [--rankings attr,relp] [--limit 20]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attn_pattern_patch import compute_positions
from patch_sweep import align_suffix_positions
# Reuse the SiLU + RMSNorm linearization for optional RelP attribution.
from mlp_sparse_patch import linearized_for_backward


K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 14336]
TAU_LEVELS = [0.0, 0.25, 0.5]   # plus implicit "final-only" baseline
TAU_NAMES  = {0.0: "all", 0.25: "tau25", 0.5: "tau50"}


def to_long_1d(x):
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
    else:
        t = torch.as_tensor(x)
    t = t.long().squeeze()
    if t.dim() == 0:
        t = t.unsqueeze(0)
    return t


def logit_diff(logits, safe_tok, unsafe_tok):
    return (logits[0, -1, safe_tok] - logits[0, -1, unsafe_tok]).item()


def make_capture_nograd(L, buf):
    def hook(module, args):
        buf[L] = args[0].detach()
        return None
    return hook


def make_capture_withgrad(L, buf):
    def hook(module, args):
        x = args[0].detach().requires_grad_(True)
        buf[L] = x
        return (x,)
    return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layers", default=",".join(str(i) for i in range(13)))
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rankings", default="attr",
                    help="attr (nonlinear AP) and/or relp (linearized).")
    ap.add_argument("--include-unsafe-unsafe", action="store_true",
                    help="Include attacks that ended unsafe (different "
                    "category) in addition to unsafe->safe.")
    args = ap.parse_args()
    rankings_to_run = [r.strip() for r in args.rankings.split(",") if r.strip()]

    pt_dir = Path(args.pt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_layers = [int(s) for s in args.layers.split(",") if s.strip()]

    pt_files = sorted(pt_dir.glob("prompt_*.pt"))
    records = []
    for p in pt_files:
        d = torch.load(p, weights_only=False)
        if "all_suffix_ids" not in d or "prompt_ids" not in d \
           or "attacked_prompt_ids" not in d:
            continue
        if args.include_unsafe_unsafe:
            keep = bool(d.get("success"))
        else:
            keep = (d.get("post_verdict") == "safe" and d.get("success"))
        if keep:
            records.append((p, d))
    if args.limit:
        records = records[: args.limit]
    print(f"Using {len(records)} attacks, layers {target_layers}, "
          f"rankings {rankings_to_run}")

    tok_path = args.model_path or records[0][1]["tokenizer_name_or_path"]
    print(f"Loading model: {tok_path}")
    tok = AutoTokenizer.from_pretrained(tok_path)
    model = AutoModelForCausalLM.from_pretrained(
        tok_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    dev = model.device
    inter_size = model.config.intermediate_size
    n_model_layers = model.config.num_hidden_layers
    print(f"  model layers={n_model_layers}, intermediate_size={inter_size}")
    for p in model.parameters():
        p.requires_grad_(False)

    NEWLINE_TOK = tok("\n\n",   add_special_tokens=False).input_ids[0]
    SAFE_TOK    = tok("safe",   add_special_tokens=False).input_ids[0]
    UNSAFE_TOK  = tok("unsafe", add_special_tokens=False).input_ids[0]
    FILLER_TOK  = tok("!",      add_special_tokens=False).input_ids[0]

    # --------------------------------------------------------------------
    # Per-prompt attribution accumulators: one per (ranking, tau, prompt, L).
    # Stored on CPU as float32 tensors of shape (inter_size,). Per-prompt
    # mean is computed at the end, then averaged across prompts.
    # --------------------------------------------------------------------
    # accum[ranking][tau_label][prompt_idx][L] = (sum_attr_d_inter, n_kept)
    TAU_LABELS = ["final"] + [TAU_NAMES[t] for t in TAU_LEVELS]
    accum = {r: {tl: [] for tl in TAU_LABELS} for r in rankings_to_run}
    # Clean activation cache (for the recovery-sweep): per prompt, per layer,
    # captured at the suffix positions of the FINAL attacked input.
    clean_acts = []          # list of dict[L] -> (n_pos, inter) cpu bf16
    prompt_meta = []

    t0 = time.time()
    for pi, (pt_path, d) in enumerate(records, start=1):
        prompt_ids   = to_long_1d(d["prompt_ids"])
        attacked_raw = to_long_1d(d["attacked_prompt_ids"])
        all_suff     = d["all_suffix_ids"]
        if not isinstance(all_suff, (list, tuple)):
            all_suff = [all_suff]
        all_suff = [to_long_1d(s) for s in all_suff]

        # Suffix positions in attacked (without the appended verdict newline)
        attacked_full = torch.cat([attacked_raw, torch.tensor([NEWLINE_TOK])]).unsqueeze(0).to(dev)
        total_len = attacked_full.shape[1]
        s_positions, _ = compute_positions(tok, prompt_ids, attacked_raw, total_len)
        if not s_positions:
            print(f"  [{pi}] {pt_path.name}: no suffix positions; skip")
            continue
        s_start, s_end = align_suffix_positions(prompt_ids, attacked_raw)
        suffix_len = s_end - s_start
        sp = torch.tensor(s_positions, device=dev, dtype=torch.long)

        # Length-matched filler clean baseline on the FINAL attacked seq
        benign_raw = attacked_raw.clone()
        benign_raw[s_start:s_end] = FILLER_TOK
        benign_full = torch.cat([benign_raw, torch.tensor([NEWLINE_TOK])]).unsqueeze(0).to(dev)

        # Capture clean activations at suffix positions
        cap = {}
        handles = [model.model.layers[L].mlp.down_proj.register_forward_pre_hook(
                       make_capture_nograd(L, cap)) for L in target_layers]
        with torch.no_grad():
            cl_logits = model(benign_full).logits
        clean_diff = logit_diff(cl_logits, SAFE_TOK, UNSAFE_TOK)
        clean_at_sp = {L: cap[L][0, sp, :].detach().cpu() for L in target_layers}
        for h in handles: h.remove()

        # Final attacked diff (for gap and for recovery measurement)
        with torch.no_grad():
            corr_logits = model(attacked_full).logits
        corr_diff = logit_diff(corr_logits, SAFE_TOK, UNSAFE_TOK)
        gap = corr_diff - clean_diff
        if abs(gap) < 1e-3:
            print(f"  [{pi}] {pt_path.name}: degenerate gap {gap:+.4f}; skip")
            continue

        # Build per-prompt (ranking, tau) -> (sum_attr[L], n_kept) dict.
        per_prompt = {r: {tl: {L: torch.zeros(inter_size, dtype=torch.float32)
                               for L in target_layers}
                          for tl in TAU_LABELS} for r in rankings_to_run}
        per_prompt_n = {r: {tl: 0 for tl in TAU_LABELS} for r in rankings_to_run}

        # Build the unique-states list with global suffix-identity dedup.
        # The FINAL attacked suffix (= best_suffix used to build
        # attacked_prompt_ids) may or may not appear verbatim in
        # all_suffix_ids — track it separately so it always lands in the
        # 'final' bucket.
        unique_states = {}      # tuple(int) -> {"ids_full":..., "is_final":bool}
        bad_lens = 0
        n_dup = 0
        for t, suf in enumerate(all_suff):
            if suf.numel() != suffix_len:
                bad_lens += 1
                continue
            key = tuple(int(x) for x in suf.tolist())
            if key in unique_states:
                n_dup += 1
                continue
            inter_raw = attacked_raw.clone()
            inter_raw[s_start:s_end] = suf
            inter_full = torch.cat([inter_raw, torch.tensor([NEWLINE_TOK])]).unsqueeze(0).to(dev)
            unique_states[key] = {"ids_full": inter_full, "is_final": False}

        final_suf_1d = attacked_raw[s_start:s_end].clone()
        final_key = tuple(int(x) for x in final_suf_1d.tolist())
        if final_key in unique_states:
            unique_states[final_key]["is_final"] = True
        else:
            unique_states[final_key] = {"ids_full": attacked_full, "is_final": True}
        steps_to_process = [
            ("final" if v["is_final"] else "step", v["ids_full"], v["is_final"])
            for v in unique_states.values()
        ]

        n_attr_passes = 0
        for label, inter_full, is_final in steps_to_process:
            # Forward to compute inter_diff (also reused for AP backward)
            cap_g = {}
            handles = [model.model.layers[L].mlp.down_proj.register_forward_pre_hook(
                           make_capture_withgrad(L, cap_g)) for L in target_layers]
            with torch.enable_grad():
                inter_logits = model(inter_full).logits
                margin_t = inter_logits[0, -1, SAFE_TOK] - inter_logits[0, -1, UNSAFE_TOK]
                margin_t.backward()
            inter_diff = float(margin_t.detach())
            for h in handles: h.remove()

            gap_frac = (inter_diff - clean_diff) / gap

            # Decide which (ranking, tau) buckets this step counts toward
            buckets = []
            if "attr" in rankings_to_run:
                if is_final:
                    buckets.append(("attr", "final"))
                for tau in TAU_LEVELS:
                    if gap_frac >= tau:
                        buckets.append(("attr", TAU_NAMES[tau]))
            if not buckets and "relp" not in rankings_to_run:
                # Free grad memory and skip
                for L in target_layers:
                    cap_g[L].grad = None
                torch.cuda.empty_cache()
                continue

            # Compute AP attribution per layer and add to chosen buckets
            if "attr" in rankings_to_run:
                for L in target_layers:
                    g = cap_g[L].grad
                    if g is None:
                        raise RuntimeError(f"No grad at L={L} on {label}")
                    inter_act_sp = cap_g[L][0, sp, :].detach().to(torch.float32).cpu()
                    grad_sp      = g[0, sp, :].detach().to(torch.float32).cpu()
                    clean_sp     = clean_at_sp[L].to(torch.float32)
                    attr_vec = ((clean_sp - inter_act_sp) * grad_sp).sum(dim=0)
                    for (rk, tl) in buckets:
                        per_prompt[rk][tl][L] += attr_vec
                for (rk, tl) in buckets:
                    per_prompt_n[rk][tl] += 1
                    n_attr_passes += 1

            # Free grad memory before next step
            for L in target_layers:
                cap_g[L].grad = None
            torch.cuda.empty_cache()

            # Optional RelP path: re-run forward+backward with linearized model
            if "relp" in rankings_to_run:
                relp_buckets = []
                if is_final:
                    relp_buckets.append("final")
                for tau in TAU_LEVELS:
                    if gap_frac >= tau:
                        relp_buckets.append(TAU_NAMES[tau])
                if relp_buckets:
                    cap_g2 = {}
                    handles = [model.model.layers[L].mlp.down_proj.register_forward_pre_hook(
                                   make_capture_withgrad(L, cap_g2)) for L in target_layers]
                    with linearized_for_backward(model), torch.enable_grad():
                        lin_logits = model(inter_full).logits
                        margin_lin = lin_logits[0, -1, SAFE_TOK] - lin_logits[0, -1, UNSAFE_TOK]
                        margin_lin.backward()
                    for h in handles: h.remove()
                    for L in target_layers:
                        g2 = cap_g2[L].grad
                        v  = cap_g2[L][0, sp, :].detach().to(torch.float32).cpu()
                        # RelP: sum_p v * grad_lin(v) over suffix positions
                        relp_vec = (v * g2[0, sp, :].detach().to(torch.float32).cpu()).sum(dim=0)
                        for tl in relp_buckets:
                            per_prompt["relp"][tl][L] += relp_vec
                    for tl in relp_buckets:
                        per_prompt_n["relp"][tl] += 1
                    for L in target_layers:
                        cap_g2[L].grad = None
                    torch.cuda.empty_cache()

        # Normalise per-prompt accumulators by n_kept and stash
        for r in rankings_to_run:
            for tl in TAU_LABELS:
                n = per_prompt_n[r][tl]
                if n == 0:
                    accum[r][tl].append(None)   # placeholder
                else:
                    norm = {L: per_prompt[r][tl][L] / n for L in target_layers}
                    accum[r][tl].append(norm)

        clean_acts.append(clean_at_sp)
        prompt_meta.append({
            "source_file": pt_path.name,
            "prompt": d.get("prompt"),
            "attacked_full": attacked_full,
            "s_positions": s_positions,
            "clean_diff":   clean_diff,
            "corr_diff":    corr_diff,
            "gap":          gap,
            "n_traj":       len(all_suff),
            "n_dup":        n_dup,
            "n_kept":       {r: dict(per_prompt_n[r]) for r in rankings_to_run},
            "n_attr_passes": n_attr_passes,
            "bad_lens":     bad_lens,
        })

        elapsed = time.time() - t0
        rate = pi / elapsed
        eta = (len(records) - pi) / rate if rate > 0 else float("inf")
        nk_attr = per_prompt_n.get("attr", per_prompt_n.get("relp"))
        print(f"  [{pi}/{len(records)}] {pt_path.name}  "
              f"clean={clean_diff:+.2f}  corr={corr_diff:+.2f}  "
              f"gap={gap:+.2f}  n_dup={n_dup}  bad_lens={bad_lens}  "
              f"kept[final/all/tau25/tau50]="
              f"{[nk_attr['final'], nk_attr['all'], nk_attr['tau25'], nk_attr['tau50']]}  "
              f"elapsed={elapsed:.0f}s  eta={eta:.0f}s",
              flush=True)

    print(f"\nAttribution sweep done in {time.time() - t0:.1f}s, "
          f"{len(prompt_meta)} prompts kept.")

    # --------------------------------------------------------------------
    # Build per-(ranking, tau, layer) global attribution = mean across
    # prompts of per-prompt mean attribution. Then rank.
    # --------------------------------------------------------------------
    rankings = {r: {tl: {} for tl in TAU_LABELS} for r in rankings_to_run}
    scores   = {r: {tl: {} for tl in TAU_LABELS} for r in rankings_to_run}

    for r in rankings_to_run:
        for tl in TAU_LABELS:
            stacks = {L: [] for L in target_layers}
            for slot in accum[r][tl]:
                if slot is None:
                    continue
                for L in target_layers:
                    stacks[L].append(slot[L])
            for L in target_layers:
                if not stacks[L]:
                    rankings[r][tl][L] = torch.arange(inter_size)
                    scores[r][tl][L]   = torch.zeros(inter_size)
                    continue
                mean_attr = torch.stack(stacks[L], dim=0).mean(dim=0)
                # AP convention from mlp_sparse_patch.py:
                #   sort ascending => most-negative first => patching toward
                #   clean decreases f(=safe-pref) most, i.e. RECOVERY toward
                #   the unsafe-classifying clean value.
                # RelP convention: sort descending (most positive => largest
                #   contribution to f).
                if r == "attr":
                    rankings[r][tl][L] = torch.argsort(mean_attr, descending=False)
                else:
                    rankings[r][tl][L] = torch.argsort(mean_attr, descending=True)
                scores[r][tl][L] = mean_attr

    # --------------------------------------------------------------------
    # Recovery sweep: patch top-K of each (ranking, tau, L) on the FINAL
    # attacked state, measure recovery. Same convention as mlp_sparse_patch.py.
    # --------------------------------------------------------------------
    patch_state = {L: None for L in target_layers}

    def make_patch(L):
        def hook(module, args):
            state = patch_state[L]
            if state is None:
                return None
            pos_t, neuron_t, clean_at_pos_neurons = state
            inp = args[0]
            new_inp = inp.clone()
            new_inp[0, pos_t[:, None], neuron_t[None, :]] = clean_at_pos_neurons
            return (new_inp,)
        return hook

    patch_handles = []
    for L in target_layers:
        h = model.model.layers[L].mlp.down_proj.register_forward_pre_hook(make_patch(L))
        patch_handles.append(h)

    recovery = {r: {tl: {L: {k: [] for k in K_VALUES} for L in target_layers}
                    for tl in TAU_LABELS} for r in rankings_to_run}
    sweep_t0 = time.time()
    for r in rankings_to_run:
        for tl in TAU_LABELS:
            for L in target_layers:
                neurons_sorted = rankings[r][tl][L].to(dev)
                for prompt_idx, meta in enumerate(prompt_meta):
                    atk_ids = meta["attacked_full"]
                    sp = torch.tensor(meta["s_positions"], device=dev, dtype=torch.long)
                    clean_full = clean_acts[prompt_idx][L].to(dev, dtype=torch.bfloat16)
                    for k in K_VALUES:
                        k_eff = min(k, inter_size)
                        neurons_k = neurons_sorted[:k_eff]
                        clean_slice = clean_full[:, neurons_k]
                        patch_state[L] = (sp, neurons_k, clean_slice)
                        try:
                            with torch.no_grad():
                                p_logits = model(atk_ids).logits
                            p_diff = logit_diff(p_logits, SAFE_TOK, UNSAFE_TOK)
                        finally:
                            patch_state[L] = None
                        gap = meta["gap"]
                        rec = (meta["corr_diff"] - p_diff) / gap if abs(gap) > 1e-6 else 0.0
                        recovery[r][tl][L][k].append(rec)
                print(f"  [recovery] r={r} tau={tl} L={L:2d}: done at "
                      f"{time.time() - sweep_t0:.0f}s", flush=True)
    for h in patch_handles:
        h.remove()

    # --------------------------------------------------------------------
    # Save
    # --------------------------------------------------------------------
    dump = {
        "n_prompts":  len(prompt_meta),
        "layers":     target_layers,
        "k_values":   K_VALUES,
        "tau_levels": TAU_LEVELS,
        "tau_labels": TAU_LABELS,
        "rankings":   rankings_to_run,
        "intermediate_size": inter_size,
        "mean_recovery": {
            r: {tl: {L: {k: float(np.mean(recovery[r][tl][L][k])) for k in K_VALUES}
                     for L in target_layers}
                for tl in TAU_LABELS}
            for r in rankings_to_run
        },
        "std_recovery": {
            r: {tl: {L: {k: float(np.std(recovery[r][tl][L][k])) for k in K_VALUES}
                     for L in target_layers}
                for tl in TAU_LABELS}
            for r in rankings_to_run
        },
        "top_neurons_by_layer": {
            r: {tl: {L: rankings[r][tl][L][:64].cpu().tolist()
                     for L in target_layers}
                for tl in TAU_LABELS}
            for r in rankings_to_run
        },
        "per_prompt": [
            {"source_file": m["source_file"],
             "clean_diff":  m["clean_diff"],
             "corr_diff":   m["corr_diff"],
             "gap":         m["gap"],
             "n_traj":      m["n_traj"],
             "n_dup":       m["n_dup"],
             "n_kept":      m["n_kept"],
             "bad_lens":    m["bad_lens"]}
            for m in prompt_meta
        ],
    }
    out_json = out_dir / "trajectory_attribution_recovery.json"
    with open(out_json, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"Wrote {out_json}")

    # --------------------------------------------------------------------
    # Plot: one subplot per layer, one curve per tau bucket (per ranking)
    # --------------------------------------------------------------------
    n = len(target_layers)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    tau_colors = {"final": "#888888", "all": "#1f77b4", "tau25": "#2ca02c", "tau50": "#d62728"}
    tau_label_str = {"final": "final-only (N=Nprompts)",
                     "all": "all unique steps (τ=0)",
                     "tau25": "τ ≥ 0.25",
                     "tau50": "τ ≥ 0.5"}

    for r in rankings_to_run:
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.2 * rows), sharey=True)
        axes = np.array(axes).flatten()
        xs = np.array(K_VALUES)
        for i, L in enumerate(target_layers):
            ax = axes[i]
            for tl in TAU_LABELS:
                mean = np.array([np.mean(recovery[r][tl][L][k]) for k in K_VALUES])
                std  = np.array([np.std(recovery[r][tl][L][k])  for k in K_VALUES])
                ax.plot(xs, mean, marker="o", markersize=4, linewidth=1.8,
                        color=tau_colors[tl], label=tau_label_str[tl])
                ax.fill_between(xs, mean - std, mean + std,
                                color=tau_colors[tl], alpha=0.10)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
            ax.axhline(1, color="gray", linewidth=0.5, linestyle=":")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("k (top-k neurons patched at suffix positions)")
            ax.set_title(f"Layer {L}", fontsize=10)
            ax.grid(True, alpha=0.3, which="both")
            if i % cols == 0:
                ax.set_ylabel("Recovery toward clean logit-diff")
            ax.legend(loc="lower right", fontsize=7)
        for j in range(n, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(
            f"Trajectory-augmented MLP attribution patching, ranking={r}\n"
            f"N_prompts={len(prompt_meta)}; per-prompt-normalised attribution; "
            f"recovery measured on FINAL attacked state",
            fontsize=11,
        )
        fig.tight_layout()
        out_png = out_dir / f"trajectory_attribution_recovery_{r}.png"
        fig.savefig(out_png, dpi=140, bbox_inches="tight")
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
