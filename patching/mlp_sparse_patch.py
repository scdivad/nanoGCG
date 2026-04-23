"""
Sparse MLP-neuron patching on Llama Guard 3.

For each target layer L, we patch only the top-k MLP neurons (the
14336-wide post-SiLU gated activations that feed into down_proj) at
suffix positions. The top-k set is chosen UNIVERSALLY across prompts:

  topk[L] = argsort_desc( mean_over_prompts( mean_over_suffix_pos(
            |post_activation_attacked[L, p, n] - post_activation_clean[L, p, n]| ) ) )[:k]

Expected to show a steep drop-off — small k (tens to hundreds out of
14336) captures most of the layer's MLP-mediated contribution to the
attack, because only a few neurons carry genuine adversarial signal.

Hook point: LlamaMLP is
    down_proj( act_fn(gate_proj(x)) * up_proj(x) )
The input to down_proj is exactly the post-activation gated neurons;
we register a pre-forward hook on down_proj and replace specific
(position, neuron) entries with the length-matched benign-clean's
activations.

Clean baseline: attacked sequence with the 20-22 adversarial tokens
replaced by '!' filler at the same positions (length-matched).

Usage:
  python patching/mlp_sparse_patch.py \
    --pt-dir results/pt_local_i-gcg \
    --out-dir patching/mlp_sparse_advbench \
    --layers 0,1,2,3,5,10
"""
import argparse
import json
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path
import sys
import types

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attn_pattern_patch import compute_positions


def _linear_silu(x):
    """SiLU forward identical to x * sigmoid(x); sigmoid is detached so the
    backward treats it as a frozen multiplicative constant (RelP-style)."""
    return x * torch.sigmoid(x).detach()


def _linear_rmsnorm_forward(self, hidden_states):
    """RMSNorm with the rsqrt-of-variance factor detached in backward."""
    input_dtype = hidden_states.dtype
    h = hidden_states.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    rsqrt = torch.rsqrt(variance + self.variance_epsilon).detach()
    h = h * rsqrt
    return self.weight * h.to(input_dtype)


@contextmanager
def linearized_for_backward(model):
    """Monkey-patch SiLU and RMSNorm throughout the model so the forward
    pass is identical but the backward pass is linear (sigmoid and rsqrt
    factors frozen). Attention softmax is not linearized here — we're
    focused on MLP-neuron attribution; softmax-linearization would matter
    more for an attention-circuit analysis."""
    saved = []

    class _FrozenSigmoidSiLU(torch.nn.Module):
        def forward(self, x):
            return _linear_silu(x)

    for i, layer in enumerate(model.model.layers):
        saved.append((
            layer.input_layernorm.forward,
            layer.post_attention_layernorm.forward,
            layer.mlp.act_fn,
        ))
        layer.input_layernorm.forward = types.MethodType(_linear_rmsnorm_forward, layer.input_layernorm)
        layer.post_attention_layernorm.forward = types.MethodType(_linear_rmsnorm_forward, layer.post_attention_layernorm)
        layer.mlp.act_fn = _FrozenSigmoidSiLU()
    saved_final_norm_forward = model.model.norm.forward
    model.model.norm.forward = types.MethodType(_linear_rmsnorm_forward, model.model.norm)
    try:
        yield
    finally:
        for i, layer in enumerate(model.model.layers):
            orig_in_ln_fwd, orig_post_ln_fwd, orig_act_fn = saved[i]
            layer.input_layernorm.forward = orig_in_ln_fwd
            layer.post_attention_layernorm.forward = orig_post_ln_fwd
            layer.mlp.act_fn = orig_act_fn
        model.model.norm.forward = saved_final_norm_forward


K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 14336]


def logit_diff(logits, safe_tok, unsafe_tok):
    return (logits[0, -1, safe_tok] - logits[0, -1, unsafe_tok]).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layers", default="0,1,2,3,5,10",
                    help="Comma-separated layer indices to analyze.")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rankings", default="diff,attr",
                    help="Comma-separated set of rankings to compare: 'diff' = "
                    "absolute activation difference; 'attr' = attribution "
                    "patching (first-order approximation "
                    "(a_C - a_R) * df/da|_R, summed over positions and prompts).")
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
        if d.get("post_verdict") == "safe" and d.get("success"):
            records.append((p, d))
    if args.limit:
        records = records[: args.limit]
    print(f"Using {len(records)} successful attacks, layers {target_layers}")

    tok_path = args.model_path or records[0][1]["tokenizer_name_or_path"]
    print(f"Loading model: {tok_path}")
    tok = AutoTokenizer.from_pretrained(tok_path)
    model = AutoModelForCausalLM.from_pretrained(
        tok_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    dev = model.device
    n_layers = model.config.num_hidden_layers
    inter_size = model.config.intermediate_size
    print(f"  intermediate_size = {inter_size}")

    NEWLINE_TOK = tok("\n\n",   add_special_tokens=False).input_ids[0]
    SAFE_TOK    = tok("safe",   add_special_tokens=False).input_ids[0]
    UNSAFE_TOK  = tok("unsafe", add_special_tokens=False).input_ids[0]
    FILLER_TOK  = tok("!",      add_special_tokens=False).input_ids[0]

    # Disable param grads to avoid backward allocating grad buffers for all
    # ~8B parameters. For attribution patching we only need grads on the
    # intermediate post-activation tensors — we achieve that by detaching
    # inside the forward_pre_hook and returning a fresh leaf tensor that
    # requires grad. The backward pass then terminates at those leaves.
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- Phase 1: capture clean + attacked post-activation neurons AND
    # gradients of logit-diff wrt attacked activations ----
    # Two gradient paths: ordinary nonlinear backward (for AP) and
    # linearized backward (for RelP).
    captured     = {L: [] for L in target_layers}    # clean per-prompt (n_pos, inter)
    attacked_acts = {L: [] for L in target_layers}    # attacked per-prompt (n_pos, inter)
    attacked_grads_nl = {L: [] for L in target_layers}  # nonlinear grad (AP)
    attacked_grads_lin = {L: [] for L in target_layers} # linearized grad (RelP)
    prompt_meta = []

    # Shared buffers: set per forward-pass.
    capture_buffer = {}           # L -> captured tensor (1, seq, inter) WITH grad retained on attacked pass

    def make_capture_nograd(L):
        def hook(module, args):
            capture_buffer[L] = args[0].detach()
            return None
        return hook

    def make_capture_withgrad(L):
        def hook(module, args):
            # Detach the composed intermediate and reintroduce it as a leaf
            # tensor with requires_grad=True. down_proj then operates on this
            # leaf, and backward from the final loss terminates at this
            # tensor — we read its gradient directly.
            x = args[0].detach().requires_grad_(True)
            capture_buffer[L] = x
            return (x,)
        return hook

    t0 = time.time()
    for i, (p, d) in enumerate(records, start=1):
        atk_raw = d["attacked_prompt_ids"]
        attacked_ids = torch.cat([atk_raw, torch.tensor([NEWLINE_TOK])]).unsqueeze(0).to(dev)
        total_len = attacked_ids.shape[1]
        s_positions, _ = compute_positions(tok, d["prompt_ids"], atk_raw, total_len)
        if not s_positions:
            continue

        benign_raw = atk_raw.clone()
        benign_raw[s_positions[0]:s_positions[-1] + 1] = FILLER_TOK
        benign_ids = torch.cat([benign_raw, torch.tensor([NEWLINE_TOK])]).unsqueeze(0).to(dev)
        sp = torch.tensor(s_positions, device=dev)

        # --- Clean forward (no grad) ---
        handles = [model.model.layers[L].mlp.down_proj.register_forward_pre_hook(make_capture_nograd(L))
                   for L in target_layers]
        capture_buffer.clear()
        with torch.no_grad():
            clean_logits = model(benign_ids).logits
        clean_diff = logit_diff(clean_logits, SAFE_TOK, UNSAFE_TOK)
        for L in target_layers:
            captured[L].append(capture_buffer[L][0, sp, :].cpu())
        for h in handles:
            h.remove()

        # --- Attacked forward + backward: nonlinear (AP grad) ---
        handles = [model.model.layers[L].mlp.down_proj.register_forward_pre_hook(make_capture_withgrad(L))
                   for L in target_layers]
        capture_buffer.clear()
        with torch.enable_grad():
            corr_logits = model(attacked_ids).logits
            loss = corr_logits[0, -1, SAFE_TOK] - corr_logits[0, -1, UNSAFE_TOK]
            loss.backward()
        corr_diff = loss.detach().item()
        for L in target_layers:
            attacked_acts[L].append(capture_buffer[L][0, sp, :].detach().cpu())
            g = capture_buffer[L].grad
            if g is None:
                raise RuntimeError(f"No gradient captured at layer {L}.")
            attacked_grads_nl[L].append(g[0, sp, :].detach().cpu())
        for h in handles:
            h.remove()
        torch.cuda.empty_cache()

        # --- Attacked forward + backward: linearized (RelP grad) ---
        if "relp" in rankings_to_run:
            handles = [model.model.layers[L].mlp.down_proj.register_forward_pre_hook(make_capture_withgrad(L))
                       for L in target_layers]
            capture_buffer.clear()
            with linearized_for_backward(model), torch.enable_grad():
                corr_logits_lin = model(attacked_ids).logits
                loss_lin = corr_logits_lin[0, -1, SAFE_TOK] - corr_logits_lin[0, -1, UNSAFE_TOK]
                loss_lin.backward()
            # Sanity: linearized forward should match nonlinear (identity in forward).
            if abs(loss_lin.item() - corr_diff) > 0.1:
                print(f"  WARN: linearized forward diverged from nonlinear: {loss_lin.item():.3f} vs {corr_diff:.3f}")
            for L in target_layers:
                g = capture_buffer[L].grad
                if g is None:
                    raise RuntimeError(f"No linearized gradient captured at layer {L}.")
                attacked_grads_lin[L].append(g[0, sp, :].detach().cpu())
            for h in handles:
                h.remove()
            torch.cuda.empty_cache()

        gap = corr_diff - clean_diff
        prompt_meta.append({
            "source_file": p.name,
            "prompt": d["prompt"],
            "attacked_ids": attacked_ids,
            "s_positions": s_positions,
            "clean_diff": clean_diff,
            "corr_diff": corr_diff,
            "gap": gap,
        })

    print(f"  captured neurons + gradients in {time.time() - t0:.1f}s")

    # ---- Phase 2: compute rankings per layer ----
    # diff ranking: argsort_desc( mean_prompts mean_pos |attacked - clean| )
    # attr ranking: argsort_asc(  mean_prompts sum_pos (clean - attacked) * (df/da|_attacked) )
    #               (ascending = most negative first = most recovery-positive when patched)
    rankings = {name: {} for name in rankings_to_run}
    scores   = {name: {} for name in rankings_to_run}

    # Prompts have variable numbers of suffix positions (20-22 due to BPE
    # boundary merges), so we aggregate per-prompt, then mean across prompts.
    for L in target_layers:
        per_prompt_diffs = []
        per_prompt_attrs = []
        per_prompt_relp  = []
        for i in range(len(captured[L])):
            ca = captured[L][i].to(torch.float32)
            aa = attacked_acts[L][i].to(torch.float32)
            ga_nl = attacked_grads_nl[L][i].to(torch.float32)
            per_prompt_diffs.append((aa - ca).abs().mean(dim=0))                  # |diff|: mean over positions
            per_prompt_attrs.append(((ca - aa) * ga_nl).sum(dim=0))               # AP: sum over positions
            if "relp" in rankings_to_run and attacked_grads_lin[L]:
                ga_lin = attacked_grads_lin[L][i].to(torch.float32)
                per_prompt_relp.append((aa * ga_lin).sum(dim=0))                  # RelP: v * grad_linearized, sum over positions

        if "diff" in rankings_to_run:
            diff = torch.stack(per_prompt_diffs, dim=0).mean(dim=0)
            rankings["diff"][L] = torch.argsort(diff, descending=True)
            scores["diff"][L]   = diff
            top5 = rankings["diff"][L][:5]
            print(f"  L={L:2d} diff top-5: idx={top5.tolist()}  score={[round(diff[i].item(), 4) for i in top5]}")

        if "attr" in rankings_to_run:
            attr = torch.stack(per_prompt_attrs, dim=0).mean(dim=0)
            rankings["attr"][L] = torch.argsort(attr, descending=False)   # most-negative first (patching decreases f)
            scores["attr"][L]   = attr
            top5 = rankings["attr"][L][:5]
            print(f"  L={L:2d} attr top-5: idx={top5.tolist()}  score={[round(attr[i].item(), 4) for i in top5]}")

        if "relp" in rankings_to_run and per_prompt_relp:
            relp = torch.stack(per_prompt_relp, dim=0).mean(dim=0)
            # RelP_v = v * d(f_linearized)/dv. Positive score = neuron
            # currently contributes positively to f (= safe-preferring).
            # Patching these toward clean should decrease f → recovery.
            # Rank by RelP score DESCENDING (most positive first).
            rankings["relp"][L] = torch.argsort(relp, descending=True)
            scores["relp"][L]   = relp
            top5 = rankings["relp"][L][:5]
            print(f"  L={L:2d} relp top-5: idx={top5.tolist()}  score={[round(relp[i].item(), 4) for i in top5]}")
    topk_order = rankings  # dict of dict: rankings[name][L] -> sorted neuron indices

    # ---- Phase 3: sweep k for each layer, measure recovery ----
    # One-shot pre-hook on each target layer's down_proj reads a
    # per-call state: (positions, top_neuron_indices, clean_slice).
    patch_state = {L: None for L in target_layers}

    def make_patch(L):
        def hook(module, args):
            state = patch_state[L]
            if state is None:
                return None
            pos_t, neuron_t, clean_at_pos_neurons = state
            inp = args[0]  # (1, seq, inter)
            new_inp = inp.clone()
            # Replace inp[0, pos_t[:, None], neuron_t[None, :]] with clean values
            # clean_at_pos_neurons has shape (n_pos, k)
            new_inp[0, pos_t[:, None], neuron_t[None, :]] = clean_at_pos_neurons
            return (new_inp,)
        return hook

    patch_handles = []
    for L in target_layers:
        h = model.model.layers[L].mlp.down_proj.register_forward_pre_hook(make_patch(L))
        patch_handles.append(h)

    # recovery[ranking_name][L][k] = list of per-prompt recoveries
    recovery = {name: {L: {k: [] for k in K_VALUES} for L in target_layers}
                for name in rankings_to_run}
    sweep_t0 = time.time()
    for ranking_name in rankings_to_run:
        for L in target_layers:
            neurons_sorted = topk_order[ranking_name][L].to(dev)
            for prompt_idx, meta in enumerate(prompt_meta):
                atk_ids = meta["attacked_ids"]
                sp = torch.tensor(meta["s_positions"], device=dev, dtype=torch.long)
                clean_full = captured[L][prompt_idx].to(dev, dtype=torch.bfloat16)  # (n_pos, inter)
                for k in K_VALUES:
                    k = min(k, inter_size)
                    neurons_k = neurons_sorted[:k]
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
                    recovery[ranking_name][L][k].append(rec)
            print(f"  [{ranking_name}] layer {L}: done in {time.time() - sweep_t0:.1f}s total")
    for h in patch_handles:
        h.remove()

    # ---- Save + plot ----
    out_json = out_dir / "mlp_sparse_recovery.json"
    dump = {
        "n_prompts": len(prompt_meta),
        "layers": target_layers,
        "k_values": K_VALUES,
        "intermediate_size": inter_size,
        "rankings": rankings_to_run,
        "mean_recovery": {
            name: {L: {k: float(np.mean(recovery[name][L][k])) for k in K_VALUES}
                   for L in target_layers}
            for name in rankings_to_run
        },
        "std_recovery": {
            name: {L: {k: float(np.std(recovery[name][L][k])) for k in K_VALUES}
                   for L in target_layers}
            for name in rankings_to_run
        },
        "top_neurons_by_layer": {
            name: {L: topk_order[name][L][:64].cpu().tolist() for L in target_layers}
            for name in rankings_to_run
        },
    }
    with open(out_json, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"Wrote {out_json}")

    # Plot: one subplot per layer, two curves (diff and attr)
    n = len(target_layers)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.2 * rows), sharey=True)
    axes = np.array(axes).flatten()
    xs = np.array(K_VALUES)
    name_colors = {"diff": "#d62728", "attr": "#1f77b4", "relp": "#2ca02c"}
    name_labels = {"diff": "|diff| rank", "attr": "AP rank", "relp": "RelP rank (linearized)"}
    for i, L in enumerate(target_layers):
        ax = axes[i]
        for name in rankings_to_run:
            mean = np.array([np.mean(recovery[name][L][k]) for k in K_VALUES])
            std  = np.array([np.std(recovery[name][L][k]) for k in K_VALUES])
            ax.plot(xs, mean, marker="o", markersize=4, linewidth=1.8,
                    color=name_colors[name], label=name_labels[name])
            ax.fill_between(xs, mean - std, mean + std, color=name_colors[name], alpha=0.12)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.axhline(1, color="gray", linewidth=0.5, linestyle=":")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k (top-k neurons patched at suffix positions)")
        ax.set_title(f"Layer {L}", fontsize=10)
        ax.grid(True, alpha=0.3, which="both")
        if i % cols == 0:
            ax.set_ylabel("Recovery toward clean logit-diff")
        ax.legend(loc="lower right", fontsize=8)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Sparse MLP-neuron patching on Llama Guard 3 — |diff| vs. attribution patching ranking\n"
        f"Patch post-SiLU gated MLP neurons at suffix positions; N={len(prompt_meta)} attacks; "
        f"intermediate size = {inter_size}",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = out_dir / "mlp_sparse_recovery.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
