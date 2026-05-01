"""
Llama Guard top-K MLP neuron interpretation: top-activations, AUROC,
MIN-z / SUM-z conjunction scan.

Mirrors cluster_neurons_individual_topact.py + cluster_neurons_polyconj.py
from the BERT-pipeline, adapted for Llama-arch (LlamaGuard-7B / Llama
Guard 3 8B). Reads the top-K (L, n) from a trajectory attribution JSON
(produced by trajectory_attribution_topk.py), runs them on a balanced
harm/safe corpus, and reports:

  (1) Per-neuron top-N activating positions across the corpus (with
      ±CTX_RADIUS token window). Reveals what each neuron individually
      fires on.

  (2) Per-neuron polysemanticity / harm-vs-safe stats: harm_mean,
      safe_mean (max-over-positions per prompt), Cohen's d, AUROC.
      AUROC ~ 0.5 means it is not a harm detector — its top-activations
      are corpus-imbalance artefacts.

  (3) MIN-z + SUM-z conjunction scan: z-score each top-K neuron across
      all (prompt, position) pairs, then rank positions by:
        - SUM-z: Sum of z across the K neurons (loose conjunction)
        - MIN-z: min of z across the K neurons (strict — every neuron
          must fire above its baseline at this position)
      Surfaces the joint concept that the K-neuron set encodes.

Hook point: model.model.layers[L].mlp.down_proj forward-pre hook
(captures the post-SiLU gated activation entering down_proj). Same as
trajectory_attribution_topk.py, so the (L, n) indices are directly
comparable.

Usage (LG1):
  python patching/lg_neurons_inspect.py \\
      --attribution-json patching/trajectory_attribution_48670390_lg1_acg/trajectory_attribution_recovery.json \\
      --model-path /home/dcheung2/new/guard_lat/LlamaGuard-7b \\
      --aegis-train /home/dcheung2/new/harm_classifiers/datasets/aegis/train \\
      --top-k 32 --n-per-class 200 --tau-bucket all
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from datasets import load_from_disk
except ImportError as e:
    raise ImportError("datasets package required (`pip install datasets`)") from e


def auroc(scores_pos, scores_neg):
    """Mann-Whitney U / |pos|·|neg|. 0.5 = no discrimination, 1.0 = perfect."""
    if not len(scores_pos) or not len(scores_neg):
        return float("nan")
    all_scores = np.concatenate([scores_pos, scores_neg])
    labels = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    sum_pos = ranks[labels == 1].sum()
    n_pos = len(scores_pos); n_neg = len(scores_neg)
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def cohen_d(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    pooled = np.sqrt((np.var(pos, ddof=1) + np.var(neg, ddof=1)) / 2)
    return float((pos.mean() - neg.mean()) / pooled) if pooled > 1e-9 else float("nan")


def context_string(tok, ids_1d, pos, win):
    lo = max(0, pos - win); hi = min(len(ids_1d), pos + win + 1)
    toks = tok.convert_ids_to_tokens(ids_1d[lo:hi].tolist())
    return " ".join(("↦" + t + "↤") if i == pos - lo else t
                    for i, t in enumerate(toks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribution-json", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--aegis-train", required=True,
                    help="Path to load_from_disk-able dataset with "
                    "{text, label} fields. Use the same training "
                    "distribution the model saw.")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write the JSON summary. Defaults to "
                    "<dirname(attribution-json)>/inspect/")
    ap.add_argument("--top-k", type=int, default=32,
                    help="Number of top-K (L, n) tuples to inspect.")
    ap.add_argument("--ranking", default="attr", choices=["attr", "relp"])
    ap.add_argument("--tau-bucket", default="all",
                    choices=["final", "all", "tau25", "tau50"])
    ap.add_argument("--n-per-class", type=int, default=200,
                    help="Balanced N harm + N safe prompts.")
    ap.add_argument("--n-top-per-neuron", type=int, default=10)
    ap.add_argument("--n-conj-top", type=int, default=15)
    ap.add_argument("--ctx-radius", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-verdict-prefix", action="store_true",
                    help="LG1 (no '\\n\\n' append). For interpretation we "
                    "don't actually score the verdict, but the chat-template"
                    " call uses the same flag for parity with attribution.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else \
              Path(args.attribution_json).parent / "inspect"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load top-K (L, n) from attribution JSON ──
    attrib = json.load(open(args.attribution_json))
    ranking_field = attrib["top_global_by_tau"][args.ranking][args.tau_bucket]
    top_neurons = [(int(L), int(n)) for L, n in ranking_field[: args.top_k]]
    print(f"Top-{args.top_k} neurons from {args.attribution_json} "
          f"(ranking={args.ranking}, tau={args.tau_bucket}):")
    for i, (L, n) in enumerate(top_neurons):
        print(f"  [{i:>3}]  L{L:>2}  #{n}")
    target_layers = sorted({L for L, _ in top_neurons})
    print(f"target_layers = {target_layers}")

    # ── Load model ──
    print(f"\nLoading model: {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    dev = model.device
    inter_size = model.config.intermediate_size
    print(f"  intermediate_size={inter_size}, n_layers={model.config.num_hidden_layers}")

    # ── Load corpus ──
    print(f"\nLoading dataset from {args.aegis_train}")
    ds = load_from_disk(args.aegis_train)
    rng = np.random.default_rng(args.seed)
    harm_all = [ex["text"] for ex in ds if ex["label"] == 1]
    safe_all = [ex["text"] for ex in ds if ex["label"] == 0]
    rng.shuffle(harm_all); rng.shuffle(safe_all)
    harm_texts = harm_all[: args.n_per_class]
    safe_texts = safe_all[: args.n_per_class]
    texts = [(t, 1) for t in harm_texts] + [(t, 0) for t in safe_texts]
    print(f"  corpus: {len(harm_texts)} harm + {len(safe_texts)} safe = {len(texts)} total")

    # ── Capture hooks ──
    capture = {}
    def make_hook(L):
        def hook(_m, args_):
            capture[L] = args_[0].detach()
            return None
        return hook

    # ── Forward each text, accumulate everything we need ──
    # For (1) per-neuron top-acts: keep best (act, label, ctx) per text per neuron
    by_neuron = {key: [] for key in top_neurons}    # list of (act_max, label, ctx_at_argmax)
    # For (2) AUROC: per-neuron list of max-act per text by class
    per_neuron_harm_max = {key: [] for key in top_neurons}
    per_neuron_safe_max = {key: [] for key in top_neurons}
    # For (3) conjunction: per-text (S, K) act matrix + ids + label
    conj_samples = []

    nt = len(top_neurons)
    t0 = time.time()
    for ti, (text, label) in enumerate(texts):
        # Apply chat template — same surface form the model sees in production.
        msg = [{"role": "user", "content": text}]
        ids = tok.apply_chat_template(
            msg, return_tensors="pt", add_generation_prompt=True,
        )
        if ids.shape[1] > args.max_len:
            ids = ids[:, -args.max_len:]
        ids = ids.to(dev)
        S = ids.shape[1]

        capture.clear()
        handles = [model.model.layers[L].mlp.down_proj.register_forward_pre_hook(make_hook(L))
                   for L in target_layers]
        try:
            with torch.no_grad():
                model(input_ids=ids)
        finally:
            for h in handles: h.remove()

        toks = tok.convert_ids_to_tokens(ids[0].cpu().tolist())
        # Build (S, nt) act matrix for conjunction
        acts_per_neuron = np.zeros((S, nt), dtype=np.float32)
        for ki, (L, n) in enumerate(top_neurons):
            v = capture[L][0, :, n].float().cpu().numpy()
            acts_per_neuron[:, ki] = v
            best_p = int(np.argmax(v))
            best_a = float(v[best_p])
            ctx = " ".join(("↦" + toks[i] + "↤") if i == best_p else toks[i]
                           for i in range(max(0, best_p - args.ctx_radius),
                                         min(S, best_p + args.ctx_radius + 1)))
            by_neuron[(L, n)].append((best_a, label, ctx))
            (per_neuron_harm_max if label == 1 else per_neuron_safe_max)[(L, n)].append(best_a)

        conj_samples.append({"label": label, "ids": ids[0].cpu(), "acts": acts_per_neuron})

        if (ti + 1) % 50 == 0 or (ti + 1) == len(texts):
            elapsed = time.time() - t0
            eta = (len(texts) - ti - 1) / (ti + 1) * elapsed
            print(f"  [{ti+1}/{len(texts)}] elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    # ── (1) Per-neuron top-N activating contexts ──
    print(f"\n{'='*72}")
    print(f"(1) Per-neuron top-{args.n_top_per_neuron} activating positions on balanced corpus")
    print(f"{'='*72}")
    topact_dump = {}
    for (L, n) in top_neurons:
        recs = sorted(by_neuron[(L, n)], key=lambda x: -x[0])
        n_harm = sum(1 for r in recs[: args.n_top_per_neuron] if r[1] == 1)
        n_safe = args.n_top_per_neuron - n_harm
        print(f"\nL{L} #{n}  top-{args.n_top_per_neuron} (HARM:{n_harm} SAFE:{n_safe})")
        rows = []
        for act, lab, ctx in recs[: args.n_top_per_neuron]:
            kind = "HARM" if lab == 1 else "SAFE"
            print(f"  act={act:+8.3f}  [{kind}]  {ctx}")
            rows.append({"act": act, "label": int(lab), "ctx": ctx})
        topact_dump[f"L{L}_n{n}"] = rows

    # ── (2) Per-neuron polysemanticity / AUROC ──
    print(f"\n{'='*72}")
    print(f"(2) Per-neuron harm-vs-safe discrimination (max-over-positions)")
    print(f"{'='*72}")
    print(f"{'L':>3} {'#neuron':>7}  {'harm_mean':>10} {'safe_mean':>10}  "
          f"{'cohen_d':>8}  {'AUROC':>7}")
    auroc_dump = []
    for (L, n) in top_neurons:
        h = np.array(per_neuron_harm_max[(L, n)])
        s = np.array(per_neuron_safe_max[(L, n)])
        au = auroc(h, s); cd = cohen_d(h, s)
        print(f"L{L:<2} {n:>7}  {h.mean():>+9.3f} {s.mean():>+9.3f}  "
              f"{cd:>+7.2f}  {au:>6.3f}")
        auroc_dump.append({"L": L, "n": n,
                           "harm_mean": float(h.mean()), "safe_mean": float(s.mean()),
                           "harm_std": float(h.std()),  "safe_std": float(s.std()),
                           "cohen_d": cd, "auroc": au})

    # ── (3) MIN-z + SUM-z conjunction ──
    print(f"\n{'='*72}")
    print(f"(3) Conjunction scan: positions where top-{nt} neurons fire together")
    print(f"{'='*72}")
    all_acts = np.concatenate([s["acts"] for s in conj_samples], axis=0)
    mu = all_acts.mean(axis=0); sd = all_acts.std(axis=0) + 1e-9
    cands_sum, cands_min = [], []
    for sample in conj_samples:
        z = (sample["acts"] - mu) / sd                  # (S, nt)
        sum_z = z.sum(axis=1); min_z = z.min(axis=1)
        for p in range(z.shape[0]):
            ctx = context_string(tok, sample["ids"], p, args.ctx_radius)
            cands_sum.append((float(sum_z[p]), int(sample["label"]), ctx))
            cands_min.append((float(min_z[p]), int(sample["label"]), ctx))
    cands_sum.sort(key=lambda x: -x[0])
    cands_min.sort(key=lambda x: -x[0])

    print(f"\nBy SUM-z (loose: most-firing across all neurons), top-{args.n_conj_top}:")
    for s, lab, ctx in cands_sum[: args.n_conj_top]:
        kind = "HARM" if lab == 1 else "SAFE"
        print(f"  sum_z={s:+8.2f}  [{kind}]  {ctx}")
    print(f"\nBy MIN-z (strict: every neuron must fire above baseline), top-{args.n_conj_top}:")
    for s, lab, ctx in cands_min[: args.n_conj_top]:
        kind = "HARM" if lab == 1 else "SAFE"
        print(f"  min_z={s:+8.2f}  [{kind}]  {ctx}")

    # ── Save ──
    summary = {
        "model_path":       args.model_path,
        "attribution_json": args.attribution_json,
        "top_k":            args.top_k,
        "ranking":          args.ranking,
        "tau_bucket":       args.tau_bucket,
        "n_per_class":      args.n_per_class,
        "top_neurons":      [{"L": L, "n": n} for L, n in top_neurons],
        "target_layers":    target_layers,
        "topact":           topact_dump,
        "auroc":            auroc_dump,
        "conj_sum_top":     [{"sum_z": s, "label": lab, "ctx": ctx}
                             for s, lab, ctx in cands_sum[: args.n_conj_top]],
        "conj_min_top":     [{"min_z": s, "label": lab, "ctx": ctx}
                             for s, lab, ctx in cands_min[: args.n_conj_top]],
    }
    out_json = out_dir / f"lg_neurons_inspect_top{args.top_k}_{args.tau_bucket}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
