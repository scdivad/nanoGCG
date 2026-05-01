"""
Llama Guard direct per-neuron harm-vs-safe AUROC, across all layers.

No attribution needed: just runs a balanced harm + safe corpus through
the model, captures the post-SiLU gated MLP activation at every (layer,
neuron, position), takes max-over-positions per (prompt, layer, neuron),
and computes AUROC for harm-vs-safe at the prompt level.

Surface the top-N neurons by AUROC across all (L, n) pairs. Compare to
the attribution-top neurons: if the two rankings are disjoint, the
attribution circuit (load-bearing under GCG attack) is NOT the same as
the natural harm-detection circuit.

Hook point matches lg_neurons_inspect.py / trajectory_attribution_topk.py
(model.model.layers[L].mlp.down_proj forward-pre hook, captures
post-SiLU gated tensor of shape (B, S, intermediate_size)).

Usage:
  python patching/lg_neurons_global_auroc.py \\
      --model-path /home/dcheung2/new/guard_lat/LlamaGuard-7b \\
      --aegis-train /home/dcheung2/new/harm_classifiers/datasets/aegis/train \\
      --out-dir patching/auroc_lg1 \\
      --layers all --n-per-class 200 --top-n 64
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from datasets import load_from_disk
except ImportError as e:
    raise ImportError("datasets package required") from e


def auroc_vectorised(harm_scores, safe_scores):
    """Vectorised Mann-Whitney across many neurons in one shot.
    harm_scores: (N_harm, K), safe_scores: (N_safe, K). Returns AUROC[K]."""
    N_h, K = harm_scores.shape
    N_s, _ = safe_scores.shape
    all_s = np.concatenate([harm_scores, safe_scores], axis=0)
    order = np.argsort(all_s, axis=0)
    ranks = np.empty_like(order, dtype=np.float64)
    rng_idx = np.broadcast_to(np.arange(1, N_h + N_s + 1)[:, None], order.shape)
    np.put_along_axis(ranks, order, rng_idx, axis=0)
    sum_pos = ranks[:N_h, :].sum(axis=0)
    return (sum_pos - N_h * (N_h + 1) / 2) / (N_h * N_s)


def cohen_d_vec(harm, safe):
    var_h = harm.var(axis=0, ddof=1)
    var_s = safe.var(axis=0, ddof=1)
    pooled = np.sqrt((var_h + var_s) / 2)
    diff = harm.mean(axis=0) - safe.mean(axis=0)
    out = np.zeros_like(diff)
    mask = pooled > 1e-9
    out[mask] = diff[mask] / pooled[mask]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--aegis-train", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layers", default="all",
                    help="Comma-separated layer indices, or 'all'.")
    ap.add_argument("--n-per-class", type=int, default=200)
    ap.add_argument("--top-n", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-verdict-prefix", action="store_true",
                    help="LG1 (no '\\n\\n' append). For activation capture "
                    "the verdict position doesn't matter, but kept for "
                    "parity with lg_neurons_inspect.py.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    dev = model.device
    inter_size = model.config.intermediate_size
    n_layers = model.config.num_hidden_layers
    print(f"  intermediate_size={inter_size}  n_layers={n_layers}")

    if args.layers == "all":
        target_layers = list(range(n_layers))
    else:
        target_layers = [int(s) for s in args.layers.split(",") if s.strip()]
    print(f"  target_layers={target_layers}")

    print(f"\nLoading dataset {args.aegis_train}")
    ds = load_from_disk(args.aegis_train)
    rng = np.random.default_rng(args.seed)
    harm_all = [ex["text"] for ex in ds if ex["label"] == 1]
    safe_all = [ex["text"] for ex in ds if ex["label"] == 0]
    rng.shuffle(harm_all); rng.shuffle(safe_all)
    harm_texts = harm_all[: args.n_per_class]
    safe_texts = safe_all[: args.n_per_class]
    texts = [(t, 1) for t in harm_texts] + [(t, 0) for t in safe_texts]
    n_total = len(texts)
    print(f"  corpus: {len(harm_texts)} harm + {len(safe_texts)} safe = {n_total}")

    n_layers_used = len(target_layers)
    L_to_idx = {L: i for i, L in enumerate(target_layers)}
    # max_per_text: (n_total, n_layers_used, inter_size) on CPU
    # For LG1 7B, 32 layers, inter=11008, N=400 -> ~564 MB. Acceptable.
    print(f"\nAllocating max_per_text: ({n_total}, {n_layers_used}, {inter_size}) "
          f"~ {n_total * n_layers_used * inter_size * 4 / 1e9:.2f} GB")
    max_per_text = np.zeros((n_total, n_layers_used, inter_size), dtype=np.float32)

    capture = {}
    def make_hook(L):
        def hook(_m, args_):
            capture[L] = args_[0].detach()
            return None
        return hook

    t0 = time.time()
    for ti, (text, _label) in enumerate(texts):
        msg = [{"role": "user", "content": text}]
        ids = tok.apply_chat_template(
            msg, return_tensors="pt", add_generation_prompt=True,
        )
        if ids.shape[1] > args.max_len:
            ids = ids[:, -args.max_len:]
        ids = ids.to(dev)

        capture.clear()
        handles = [model.model.layers[L].mlp.down_proj.register_forward_pre_hook(make_hook(L))
                   for L in target_layers]
        try:
            with torch.no_grad():
                model(input_ids=ids)
        finally:
            for h in handles: h.remove()

        for L in target_layers:
            v = capture[L][0]
            mx = v.float().max(dim=0).values.cpu().numpy()
            max_per_text[ti, L_to_idx[L], :] = mx
        capture.clear()

        if (ti + 1) % 50 == 0 or (ti + 1) == n_total:
            elapsed = time.time() - t0
            eta = (n_total - ti - 1) / (ti + 1) * elapsed
            print(f"  [{ti+1}/{n_total}]  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    labels = np.array([lab for _, lab in texts])
    harm_idx = labels == 1
    safe_idx = labels == 0
    print(f"\nComputing AUROC + Cohen's d for {n_layers_used * inter_size} (L, n) pairs...")
    aurocs = np.zeros((n_layers_used, inter_size), dtype=np.float32)
    cohen_ds = np.zeros((n_layers_used, inter_size), dtype=np.float32)
    for li, L in enumerate(target_layers):
        h = max_per_text[harm_idx, li, :]
        s = max_per_text[safe_idx, li, :]
        aurocs[li, :] = auroc_vectorised(h, s)
        cohen_ds[li, :] = cohen_d_vec(h, s)
    print(f"  done ({time.time() - t0:.0f}s total)")

    # Top-N global by |AUROC - 0.5|
    flat_au = aurocs.flatten()
    flat_d = cohen_ds.flatten()
    distance = np.abs(flat_au - 0.5)
    order = np.argsort(-distance)

    print(f"\n{'='*72}")
    print(f"Top-{args.top_n} (L, n) by |AUROC - 0.5| (most discriminative either direction)")
    print(f"{'='*72}")
    print(f"  {'rank':>4}  {'L':>3} {'#neuron':>7}  {'AUROC':>7}  {'cohen_d':>8}  "
          f"{'harm_mean':>10}  {'safe_mean':>10}")
    rows = []
    for r, idx in enumerate(order[: args.top_n]):
        li = idx // inter_size; ni = idx % inter_size
        L = target_layers[li]
        au = float(aurocs[li, ni]); cd = float(cohen_ds[li, ni])
        h = max_per_text[harm_idx, li, ni]
        s = max_per_text[safe_idx, li, ni]
        hm = float(h.mean()); sm = float(s.mean())
        print(f"  {r:>4}  L{L:>2} {ni:>7}  {au:>6.3f}  {cd:>+7.2f}  "
              f"{hm:>+9.3f}  {sm:>+9.3f}")
        rows.append({"rank": r, "L": L, "n": ni, "auroc": au, "cohen_d": cd,
                     "harm_mean_max": hm, "safe_mean_max": sm})

    # Per-layer summary
    print(f"\n{'='*72}")
    print(f"Per-layer top-AUROC summary")
    print(f"{'='*72}")
    print(f"  {'L':>3}  {'#high (>0.7 or <0.3)':>20}  {'#mid (.6-.7 or .3-.4)':>22}  "
          f"{'best_n':>7}  {'best_auroc':>10}  {'best_d':>7}")
    layer_summary = []
    for li, L in enumerate(target_layers):
        au_l = aurocs[li, :]
        cd_l = cohen_ds[li, :]
        n_high = int((au_l > 0.7).sum() + (au_l < 0.3).sum())
        n_mid = int(((au_l > 0.6) & (au_l <= 0.7)).sum() +
                    ((au_l < 0.4) & (au_l >= 0.3)).sum())
        best_idx = int(np.argmax(np.abs(au_l - 0.5)))
        print(f"  L{L:>2}  {n_high:>20}  {n_mid:>22}  {best_idx:>7}  "
              f"{au_l[best_idx]:>9.3f}  {cd_l[best_idx]:>+6.2f}")
        layer_summary.append({"L": L, "n_high": n_high, "n_mid": n_mid,
                              "best_n": best_idx,
                              "best_auroc": float(au_l[best_idx]),
                              "best_cohen_d": float(cd_l[best_idx])})

    out_csv = out_dir / "global_auroc_topN.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "L", "n", "auroc", "cohen_d",
                                          "harm_mean_max", "safe_mean_max"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv}")

    out_csv2 = out_dir / "global_auroc_per_layer.csv"
    with open(out_csv2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["L", "n_high", "n_mid", "best_n",
                                          "best_auroc", "best_cohen_d"])
        w.writeheader()
        w.writerows(layer_summary)
    print(f"Wrote {out_csv2}")

    np.savez_compressed(out_dir / "global_auroc_matrix.npz",
                        target_layers=np.array(target_layers, dtype=np.int32),
                        auroc=aurocs, cohen_d=cohen_ds)
    print(f"Wrote {out_dir / 'global_auroc_matrix.npz'}")


if __name__ == "__main__":
    main()
