"""
Activation patching sweep on Llama Guard 3, aggregated across multiple
successful attacks.

For each .pt file in --pt-dir whose attack was successful (post_verdict ==
"safe"):
  For each component c in {q_proj, k_proj, v_proj, o_proj, mlp}, each
  layer L in 0..31, each position-set P in {suffix, content, last}:
    * Run attacked forward with a hook that replaces component c's output
      at layer L at the positions in P with the length-matched benign-
      clean's activation (same length as attacked, '!' at the adversarial
      suffix positions).
    * Measure safe-vs-unsafe logit diff at the final position and compute
      recovery = (corr_diff - patched_diff) / (corr_diff - clean_diff).

Then average recovery across prompts: mean ± 1 std, plotted per component
for each position set. Each position set gets its own subplot.

Usage:
  python patching/patch_sweep.py \
      --pt-dir results/pt_local_i-gcg \
      --out-dir patching
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer


COMPONENTS = ["q_proj", "k_proj", "v_proj", "o_proj", "mlp"]
COMPONENT_COLORS = {
    "q_proj": "#1f77b4",
    "k_proj": "#ff7f0e",
    "v_proj": "#2ca02c",
    "o_proj": "#d62728",
    "mlp":    "#9467bd",
}
# 5-way position split (finer than suffix/content/last):
#   scaffold_header  - fixed template prefix + safety taxonomy (~175 toks, identical across prompts)
#   user_content     - the actual prompt text (varies per prompt; small for AdvBench, large for Aegis)
#   adversarial      - GCG-optimized tokens (formerly "suffix"; also covers prefix-mode attacks)
#   scaffold_tail    - fixed template tail: '\n\n<END CONVERSATION>' + assessment instructions + assistant header
#   last             - the appended '\n\n' where we read off the verdict logits
PSETS = ["scaffold_header", "user_content", "adversarial", "scaffold_tail", "last"]


def _lcp(a, b):
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return i


def _lct(a, b, lcp):
    j = 0
    while j < len(a) - lcp and j < len(b) - lcp and a[-1 - j] == b[-1 - j]:
        j += 1
    return j


def align_suffix_positions(prompt_ids, attacked_ids):
    """Return (suffix_start, suffix_end) in attacked_ids by longest common
    prefix/tail alignment against the shorter unattacked prompt_ids."""
    a = prompt_ids.tolist() if hasattr(prompt_ids, "tolist") else list(prompt_ids)
    b = attacked_ids.tolist() if hasattr(attacked_ids, "tolist") else list(attacked_ids)
    i = _lcp(a, b)
    j = _lct(a, b, i)
    return i, len(b) - j


def compute_position_sets(tokenizer, prompt_ids, attacked_ids, total_len):
    """Build the 5-way position split for the attacked input (including the
    appended final '\\n\\n' at index total_len-1).

    Buckets are CONTIGUOUS and cover [0, total_len) exactly. This avoids
    the orphan-boundary-token bug where a BPE-merged token between
    scaffold_header/user_content/adv/scaffold_tail falls into no bucket
    and leaks attack signal through c->c attention (c->s was blocked but
    the orphan sits in neither set).

    Derivation:
      header_len via LCP(empty_template, clean) — template prefix is
        identical in clean and attacked.
      user_len   via empty/clean alignment (len(clean) - header_len - tail_len).
      adv_start, adv_end via LCP/LCT(clean, attacked) — ground-truth adv
        region including any BPE-merged boundary tokens.
      mode detected by where adv sits relative to header.
      scaffold_tail absorbs any orphan boundary positions between adv and
        the appended final newline; user_content length is taken from
        clean (stable) and placed adjacent to adv.
    """
    empty_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        return_tensors="pt", add_generation_prompt=True,
    )[0].tolist()
    clean = prompt_ids.tolist() if hasattr(prompt_ids, "tolist") else list(prompt_ids)
    header_len = _lcp(empty_ids, clean)
    empty_tail_len = _lct(empty_ids, clean, header_len)
    user_len = len(clean) - header_len - empty_tail_len

    adv_start, adv_end = align_suffix_positions(prompt_ids, attacked_ids)
    last_pos = total_len - 1

    # prefix_mode when adv sits right after the header (allow 1-token drift)
    prefix_mode = abs(adv_start - header_len) <= 1

    if prefix_mode:
        header_positions = list(range(0, adv_start))
        adv_positions    = list(range(adv_start, adv_end))
        user_positions   = list(range(adv_end, adv_end + user_len))
        tail_positions   = list(range(adv_end + user_len, last_pos))
    else:
        header_positions = list(range(0, header_len))
        user_positions   = list(range(header_len, adv_start))
        adv_positions    = list(range(adv_start, adv_end))
        tail_positions   = list(range(adv_end, last_pos))
    last_positions = [last_pos]

    return {
        "scaffold_header": header_positions,
        "user_content":    user_positions,
        "adversarial":     adv_positions,
        "scaffold_tail":   tail_positions,
        "last":            last_positions,
    }


def sweep_one(model, tok, d, layer_indices, filler_tok, newline_tok, safe_tok, unsafe_tok, patch_mode: str = "clean"):
    """Run the 5 x L x 3 sweep on one .pt record. Returns dict keyed by
    (comp, pset) -> list of length n_layers of recovery floats, plus
    metadata (clean_diff, corr_diff, gap, suffix_start, suffix_end)."""
    n_layers = model.config.num_hidden_layers
    dev = model.device

    # Align and build benign-filler clean (at adversarial positions only)
    s_start, s_end = align_suffix_positions(d["prompt_ids"], d["attacked_prompt_ids"])
    benign_raw = d["attacked_prompt_ids"].clone()
    benign_raw[s_start:s_end] = filler_tok

    # Append '\n\n' so the final position predicts the verdict token
    benign_ids   = torch.cat([benign_raw,               torch.tensor([newline_tok])]).unsqueeze(0).to(dev)
    attacked_ids = torch.cat([d["attacked_prompt_ids"], torch.tensor([newline_tok])]).unsqueeze(0).to(dev)
    total_len = attacked_ids.shape[1]

    # 5-way position split
    pset_positions = compute_position_sets(
        tok, d["prompt_ids"], d["attacked_prompt_ids"], total_len
    )

    # Discover the decoder-layer list. PeftModel wraps the HF CausalLM, so
    # the layers are one hop deeper: base_model.model.model.layers vs.
    # plain model.model.layers.
    try:
        _layers = model.base_model.model.model.layers  # PeftModel
    except AttributeError:
        _layers = model.model.layers                   # plain HF CausalLM

    def get_module(layer_idx, comp):
        block = _layers[layer_idx]
        return block.mlp if comp == "mlp" else getattr(block.self_attn, comp)

    def logit_diff(logits):
        return (logits[0, -1, safe_tok] - logits[0, -1, unsafe_tok]).item()

    # Capture all clean activations in one forward pass
    captured = {}
    def make_capture(L, C):
        def hook(m, i, o):
            captured[(L, C)] = o.detach().clone()
        return hook
    hooks = []
    for L in range(n_layers):
        for C in COMPONENTS:
            hooks.append(get_module(L, C).register_forward_hook(make_capture(L, C)))
    with torch.no_grad():
        clean_logits = model(benign_ids).logits
    for h in hooks:
        h.remove()
    clean_diff = logit_diff(clean_logits)

    with torch.no_grad():
        corr_logits = model(attacked_ids).logits
    corr_diff = logit_diff(corr_logits)
    gap = corr_diff - clean_diff

    if abs(gap) < 1e-6:
        # Degenerate: clean and attacked agree. Recovery undefined; skip.
        return None

    def make_patch(positions, clean_tensor):
        pos_tensor = torch.tensor(positions, device=dev, dtype=torch.long)
        if patch_mode == "zero":
            # Zero-ablation: drop the module's output at those positions to 0.
            # For Q/K/V this wipes the token's participation in attention
            # along that pathway (e.g. zero K means attn_logits[:, j] = 0
            # for j in positions, so softmax distributes uniformly among
            # those positions regardless of the query).
            def hook(m, i, o):
                new = o.clone()
                new[:, pos_tensor, :] = 0
                return new
        else:  # "clean" — patch to length-matched benign-clean activation
            def hook(m, i, o):
                new = o.clone()
                new[:, pos_tensor, :] = clean_tensor[:, pos_tensor, :]
                return new
        return hook

    recovery = {c: {p: [] for p in PSETS} for c in COMPONENTS}
    for L in layer_indices:
        for C in COMPONENTS:
            clean_act = captured[(L, C)]
            for pset_name in PSETS:
                positions = pset_positions[pset_name]
                if not positions:
                    recovery[C][pset_name].append(0.0)
                    continue
                h = get_module(L, C).register_forward_hook(make_patch(positions, clean_act))
                with torch.no_grad():
                    p_logits = model(attacked_ids).logits
                h.remove()
                rec = (corr_diff - logit_diff(p_logits)) / gap
                recovery[C][pset_name].append(rec)

    return {
        "clean_diff":    clean_diff,
        "corr_diff":     corr_diff,
        "gap":           gap,
        "suffix_start":  s_start,
        "suffix_end":    s_end,
        "last_pos":      total_len - 1,
        "pset_sizes":    {k: len(v) for k, v in pset_positions.items()},
        "recovery":      recovery,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-dir", required=True, help="Directory of .pt files from llama_guard.py")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent), help="Where to write plot + json")
    ap.add_argument("--model-path", default=None, help="Override tokenizer_name_or_path from .pt")
    ap.add_argument("--limit", type=int, default=None, help="Cap number of successful attacks used")
    ap.add_argument("--patch-mode", choices=["clean", "zero"], default="clean",
                    help="'clean' = patch to length-matched benign-clean activation; "
                    "'zero' = zero-ablate the module's output at the selected positions.")
    ap.add_argument("--adapter-path", default=None,
                    help="If set, attach a PEFT (e.g. LoRA) adapter to the base "
                    "model via PeftModel.from_pretrained. Mirrors the loading "
                    "path used by examples/llama_guard.py when generating LAT "
                    "attacks — activation hooks will see the adapter's "
                    "modified forward (base + LoRA deltas).")
    args = ap.parse_args()

    pt_dir = Path(args.pt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate + filter to successful attacks
    pt_files = sorted(pt_dir.glob("prompt_*.pt"))
    print(f"Found {len(pt_files)} .pt files in {pt_dir}")
    records = []
    for p in pt_files:
        d = torch.load(p, weights_only=False)
        if d.get("post_verdict") == "safe" and d.get("success"):
            records.append((p, d))
        else:
            print(f"  SKIP {p.name}  post={d.get('post_verdict')}  success={d.get('success')}")
    print(f"Using {len(records)} successful attacks")
    if args.limit:
        records = records[: args.limit]
        print(f"Capped to {len(records)} (--limit)")
    if not records:
        raise SystemExit("No successful attacks found. Run llama_guard.py with --pt-output-dir first.")

    # Load model once (from the first record's tokenizer path, or override).
    tok_path = args.model_path or records[0][1]["tokenizer_name_or_path"]
    print(f"Loading model: {tok_path}")
    tok = AutoTokenizer.from_pretrained(tok_path)
    model = AutoModelForCausalLM.from_pretrained(
        tok_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    if args.adapter_path:
        from peft import PeftModel
        print(f"Attaching adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model.eval()
    print(f"Model ready ({'LAT/adapter' if args.adapter_path else 'base'}).")
    # Peft wraps model; the config (and layer list) lives on model.base_model
    # for PeftModel. We try both — n_layers needs the underlying transformer's config.
    root = model.base_model.model if args.adapter_path else model
    n_layers = root.config.num_hidden_layers
    print(f"  layers = {n_layers}")

    NEWLINE_TOK = tok("\n\n",   add_special_tokens=False).input_ids[0]
    SAFE_TOK    = tok("safe",   add_special_tokens=False).input_ids[0]
    UNSAFE_TOK  = tok("unsafe", add_special_tokens=False).input_ids[0]
    FILLER_TOK  = tok("!",      add_special_tokens=False).input_ids[0]
    assert tok.decode([FILLER_TOK]) == "!"
    layer_indices = list(range(n_layers))

    # Per-prompt sweeps
    all_results = []
    t0 = time.time()
    for i, (pt_path, d) in enumerate(records, start=1):
        print(f"\n[{i}/{len(records)}] {pt_path.name}  prompt={d['prompt']!r}")
        out = sweep_one(model, tok, d, layer_indices, FILLER_TOK, NEWLINE_TOK, SAFE_TOK, UNSAFE_TOK, patch_mode=args.patch_mode)
        if out is None:
            print("  degenerate gap; skipped")
            continue
        sizes = out['pset_sizes']
        print(f"  clean={out['clean_diff']:+.3f}  attacked={out['corr_diff']:+.3f}  "
              f"gap={out['gap']:+.3f}  adv=[{out['suffix_start']},{out['suffix_end']})  "
              f"header={sizes['scaffold_header']}  user={sizes['user_content']}  "
              f"adv={sizes['adversarial']}  tail={sizes['scaffold_tail']}  last={sizes['last']}")
        out["prompt"] = d["prompt"]
        out["best_suffix"] = d["best_suffix"]
        out["source_file"] = pt_path.name
        all_results.append(out)
    print(f"\nSwept {len(all_results)} prompts in {time.time() - t0:.1f}s")

    if not all_results:
        raise SystemExit("Every successful attack ended up with degenerate gap. Aborting.")

    # Stack recovery into arrays of shape (N_prompts, n_layers) per (comp, pset)
    N = len(all_results)
    agg_mean = {c: {p: np.zeros(n_layers) for p in PSETS} for c in COMPONENTS}
    agg_std  = {c: {p: np.zeros(n_layers) for p in PSETS} for c in COMPONENTS}
    for C in COMPONENTS:
        for P in PSETS:
            arr = np.array([r["recovery"][C][P] for r in all_results])  # (N, n_layers)
            agg_mean[C][P] = arr.mean(axis=0)
            agg_std[C][P]  = arr.std(axis=0)

    # Save numbers
    out_json = out_dir / "patching_recovery.json"
    dump = {
        "n_prompts": N,
        "n_layers":  n_layers,
        "prompts":   [r["prompt"]      for r in all_results],
        "sources":   [r["source_file"] for r in all_results],
        "per_prompt": [
            {
                "prompt":      r["prompt"],
                "source_file": r["source_file"],
                "clean_diff":  r["clean_diff"],
                "corr_diff":   r["corr_diff"],
                "gap":         r["gap"],
                "suffix_range": [r["suffix_start"], r["suffix_end"]],
                "pset_sizes":  r["pset_sizes"],
                "recovery":    r["recovery"],
            } for r in all_results
        ],
        "mean": {C: {P: agg_mean[C][P].tolist() for P in PSETS} for C in COMPONENTS},
        "std":  {C: {P: agg_std[C][P].tolist()  for P in PSETS} for C in COMPONENTS},
    }
    with open(out_json, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"Wrote {out_json}")

    # Plot: one subplot per position set, mean ± 1 std bands
    n_psets = len(PSETS)
    fig, axes = plt.subplots(1, n_psets, figsize=(4 * n_psets + 2, 5), sharey=True)
    if n_psets == 1:
        axes = [axes]
    xs = np.arange(n_layers)
    for ax, pset_name in zip(axes, PSETS):
        for comp in COMPONENTS:
            m = agg_mean[comp][pset_name]
            s = agg_std[comp][pset_name]
            color = COMPONENT_COLORS[comp]
            ax.plot(xs, m, color=color, marker="o", markersize=3, linewidth=1.6, label=comp)
            ax.fill_between(xs, m - s, m + s, color=color, alpha=0.15)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.axhline(1, color="gray", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Layer (L)")
        # Position-set sizes can vary slightly per prompt (boundary BPE drift
        # and varying prompt lengths). Report the range.
        sizes = [r["pset_sizes"][pset_name] for r in all_results]
        size_lo, size_hi = min(sizes), max(sizes)
        size_str = f"{size_lo}" if size_lo == size_hi else f"{size_lo}-{size_hi}"
        ax.set_title(f"{pset_name}  ({size_str} toks)", fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Recovery toward clean logit-diff")
    axes[0].legend(loc="best", fontsize=8)

    baseline_desc = ("Clean = length-matched '!'-filler baseline."
                     if args.patch_mode == "clean"
                     else "Zero-patch = activation replaced with 0 at selected positions.")
    fig.suptitle(
        f"Llama Guard 3 activation patching ({args.patch_mode}), averaged across N={N} successful attacks\n"
        f"{baseline_desc}  Shaded = ±1 std.",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = out_dir / "patching_recovery.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
