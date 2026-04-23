"""
Attention-pattern zero-ablation on Llama Guard 3.

For each layer L in 0..31 and each bucket pair P in
{c->s, s->c, c->c, s->s}:
  - Add -inf to attn_logits[:, :, q in P.queries, k in P.keys] at layer L
    only, before softmax. Equivalent to "pretend these keys don't exist
    for those queries": those attention weights go to 0 and the rest
    renormalize.
  - Forward attacked_ids, compute safe-vs-unsafe logit-diff recovery.

Buckets:
  s = adversarial positions (20-22 toks, GCG-optimized)
  c = scaffold_tail positions (~54 fixed toks: "\\n\\n<END CONVERSATION>\\n\\n
       Provide your safety assessment..." + <|eot|> + assistant header)

Causal-mask note: in suffix mode, c is after s (tail > adv), so c->s is
meaningful (tail queries see adv keys) but s->c is not (adv queries can't
see tail keys — q<k is causally masked regardless of our intervention).
In prefix mode, same asymmetry (c still after s). So expect s->c recovery
to be flat ~0 in both datasets — it's already structurally zero.

Attention implementation: forced to "eager" (not SDPA/Flash) so the
forward computes explicit softmax and we can add to the attention_mask
that's passed in.

Hook mechanism: one forward_pre_hook per self_attn module that reads the
module's `_custom_attn_mask` attribute and adds it to the attention_mask
kwarg. We set/clear this attribute per-layer per-patch.

Usage:
  python patching/attn_pattern_patch.py \
    --pt-dir results/pt_local_i-gcg \
    --out-dir patching/attn_pattern_suffix_advbench
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


PAIR_COLORS = {
    "c->s": "#d62728",
    "s->c": "#1f77b4",
    "c->c": "#2ca02c",
    "s->s": "#9467bd",
}
PAIR_LABELS = {
    "c->s": "c→s  (scaffold_tail queries,  adversarial keys)",
    "s->c": "s→c  (adversarial queries,    scaffold_tail keys)",
    "c->c": "c→c  (scaffold_tail queries,  scaffold_tail keys)",
    "s->s": "s→s  (adversarial queries,    adversarial keys)",
}


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


def compute_positions(tokenizer, prompt_ids, attacked_ids, total_len):
    """Identify adversarial (s) and scaffold_tail (c) positions in the
    attacked input (total_len = len(attacked_ids) + 1 for the appended
    verdict newline).

    Buckets must be CONTIGUOUS and cover [0, total_len) exactly — otherwise
    a BPE-merged boundary token between s and c can end up in neither
    bucket, freely attend to s, and then leak attack signal into c via
    c->c attention (unblocked), defeating the c->s intervention.

    Derivation:
      * header_len from LCP(empty_template, clean) — tokens are identical
        in clean and attacked template prefixes.
      * adv_start, adv_end from LCP/LCT(clean, attacked) — this is the
        only way to get adv endpoints that reflect actual token-level
        differences (including BPE boundary merges at either end).
      * In SUFFIX mode: [header | user_content | adv | scaffold_tail | last]
        with user_content = [header_end, adv_start) and
        scaffold_tail = [adv_end, last_pos). Any boundary BPE drift on
        either side gets absorbed into the adjacent non-adv bucket — no
        orphans. If the last adv token got merged with '\\n\\n' into a
        new BPE token, that merged token stays in s (via LCT).
      * In PREFIX mode: [header | adv | user_content | scaffold_tail | last].
    """
    empty_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        return_tensors="pt", add_generation_prompt=True,
    )[0].tolist()
    clean = prompt_ids.tolist() if hasattr(prompt_ids, "tolist") else list(prompt_ids)
    atk = attacked_ids.tolist() if hasattr(attacked_ids, "tolist") else list(attacked_ids)
    header_len = _lcp(empty_ids, clean)
    user_len = len(clean) - header_len - _lct(empty_ids, clean, header_len)
    adv_start = _lcp(clean, atk)
    adv_end   = len(atk) - _lct(clean, atk, adv_start)

    # Detect prefix vs. suffix mode (allow up to 1-token BPE drift).
    prefix_mode = abs(adv_start - header_len) <= 1

    last_pos = total_len - 1
    if prefix_mode:
        # adv at start of content, user_content after, tail up to the
        # appended verdict newline.
        s_positions = list(range(adv_start, adv_end))
        c_positions = list(range(adv_end + user_len, last_pos))
    else:
        # suffix mode: anything from header_end up to adv_start is
        # user_content; adv; then everything after adv up to the
        # appended verdict newline is scaffold_tail (absorbs any orphan
        # boundary tokens).
        s_positions = list(range(adv_start, adv_end))
        c_positions = list(range(adv_end, last_pos))
    return s_positions, c_positions


def install_attn_mask_hook(model):
    """Register one forward_pre_hook per self_attn that reads the module's
    `_custom_attn_mask_spec = (q_positions, k_positions)` attribute and
    adds -inf to the attention_mask kwarg at those (q, k) pairs. Builds
    the extra mask inside the hook to match the existing mask's shape
    (HF's internal causal mask may have seq_k slightly larger than
    seq_q due to cache bookkeeping)."""
    def make_hook():
        def hook(module, args, kwargs):
            spec = getattr(module, "_custom_attn_mask_spec", None)
            if spec is None:
                return args, kwargs
            q_positions, k_positions = spec
            existing = kwargs.get("attention_mask", None)
            if existing is None or not q_positions or not k_positions:
                return args, kwargs
            # existing shape: (B, 1, seq_q, seq_k)
            seq_q = existing.shape[-2]
            seq_k = existing.shape[-1]
            extra = torch.zeros_like(existing)
            q_t = torch.tensor(q_positions, device=existing.device, dtype=torch.long)
            k_t = torch.tensor(k_positions, device=existing.device, dtype=torch.long)
            q_t = q_t[q_t < seq_q]
            k_t = k_t[k_t < seq_k]
            if q_t.numel() == 0 or k_t.numel() == 0:
                return args, kwargs
            neg = torch.finfo(existing.dtype).min
            extra[:, :, q_t[:, None], k_t[None, :]] = neg
            kwargs["attention_mask"] = existing + extra
            return args, kwargs
        return hook

    handles = []
    for block in model.model.layers:
        handles.append(block.self_attn.register_forward_pre_hook(make_hook(), with_kwargs=True))
    return handles


def logit_diff(logits, safe_tok, unsafe_tok):
    return (logits[0, -1, safe_tok] - logits[0, -1, unsafe_tok]).item()


def sweep_one(model, tok, d, newline_tok, filler_tok, safe_tok, unsafe_tok):
    dev = model.device
    n_layers = model.config.num_hidden_layers

    # Build inputs
    s_start_raw, s_end_raw = None, None  # recompute inside compute_positions
    atk_raw = d["attacked_prompt_ids"]
    clean_raw = d["prompt_ids"]

    attacked_ids = torch.cat([atk_raw, torch.tensor([newline_tok])]).unsqueeze(0).to(dev)
    total_len = attacked_ids.shape[1]

    s_positions, c_positions = compute_positions(tok, clean_raw, atk_raw, total_len)
    if not s_positions or not c_positions:
        return None

    # Attacked baseline (no intervention)
    with torch.no_grad():
        corr_logits = model(attacked_ids).logits
    corr_diff = logit_diff(corr_logits, safe_tok, unsafe_tok)

    # Clean baseline: replace adversarial tokens with '!' filler, same length.
    benign_raw = atk_raw.clone()
    benign_raw[s_positions[0]:s_positions[-1] + 1] = filler_tok
    benign_ids = torch.cat([benign_raw, torch.tensor([newline_tok])]).unsqueeze(0).to(dev)
    with torch.no_grad():
        clean_logits = model(benign_ids).logits
    clean_diff = logit_diff(clean_logits, safe_tok, unsafe_tok)
    gap = corr_diff - clean_diff
    if abs(gap) < 1e-6:
        return None

    # Bucket pairs: (query_positions, key_positions)
    pair_positions = {
        "c->s": (c_positions, s_positions),
        "s->c": (s_positions, c_positions),
        "c->c": (c_positions, c_positions),
        "s->s": (s_positions, s_positions),
    }

    recovery = {name: [] for name in pair_positions}
    for L in range(n_layers):
        layer_attn = model.model.layers[L].self_attn
        for pair_name, (q_pos, k_pos) in pair_positions.items():
            layer_attn._custom_attn_mask_spec = (q_pos, k_pos)
            try:
                with torch.no_grad():
                    p_logits = model(attacked_ids).logits
                p_diff = logit_diff(p_logits, safe_tok, unsafe_tok)
                recovery[pair_name].append((corr_diff - p_diff) / gap)
            finally:
                layer_attn._custom_attn_mask_spec = None
    return {
        "clean_diff":  clean_diff,
        "corr_diff":   corr_diff,
        "gap":         gap,
        "n_s":         len(s_positions),
        "n_c":         len(c_positions),
        "recovery":    recovery,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    pt_dir = Path(args.pt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(pt_dir.glob("prompt_*.pt"))
    records = []
    for p in pt_files:
        d = torch.load(p, weights_only=False)
        if d.get("post_verdict") == "safe" and d.get("success"):
            records.append((p, d))
        else:
            print(f"  SKIP {p.name}  post={d.get('post_verdict')}")
    if args.limit:
        records = records[: args.limit]
    print(f"Using {len(records)} successful attacks")
    if not records:
        raise SystemExit("No successful attacks found.")

    tok_path = args.model_path or records[0][1]["tokenizer_name_or_path"]
    print(f"Loading model (attn_implementation=eager): {tok_path}")
    tok = AutoTokenizer.from_pretrained(tok_path)
    model = AutoModelForCausalLM.from_pretrained(
        tok_path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers

    NEWLINE_TOK = tok("\n\n",   add_special_tokens=False).input_ids[0]
    SAFE_TOK    = tok("safe",   add_special_tokens=False).input_ids[0]
    UNSAFE_TOK  = tok("unsafe", add_special_tokens=False).input_ids[0]
    FILLER_TOK  = tok("!",      add_special_tokens=False).input_ids[0]

    install_attn_mask_hook(model)

    all_results = []
    t0 = time.time()
    for i, (p, d) in enumerate(records, start=1):
        print(f"[{i}/{len(records)}] {p.name}")
        out = sweep_one(model, tok, d, NEWLINE_TOK, FILLER_TOK, SAFE_TOK, UNSAFE_TOK)
        if out is None:
            print("  skipped (degenerate gap or missing positions)")
            continue
        print(f"  clean={out['clean_diff']:+.3f}  attacked={out['corr_diff']:+.3f}  "
              f"gap={out['gap']:+.3f}  |s|={out['n_s']}  |c|={out['n_c']}")
        out["prompt"] = d["prompt"]
        out["source_file"] = p.name
        all_results.append(out)
    print(f"Swept {len(all_results)} prompts in {time.time() - t0:.1f}s")

    # Aggregate
    N = len(all_results)
    PAIRS = ["c->s", "s->c", "c->c", "s->s"]
    mean = {p: np.array([r["recovery"][p] for r in all_results]).mean(axis=0) for p in PAIRS}
    std  = {p: np.array([r["recovery"][p] for r in all_results]).std(axis=0)  for p in PAIRS}

    out_json = out_dir / "attn_pattern_recovery.json"
    with open(out_json, "w") as f:
        json.dump({
            "n_prompts": N,
            "n_layers": n_layers,
            "per_prompt": [
                {"prompt": r["prompt"], "source_file": r["source_file"],
                 "clean_diff": r["clean_diff"], "corr_diff": r["corr_diff"],
                 "gap": r["gap"], "n_s": r["n_s"], "n_c": r["n_c"],
                 "recovery": r["recovery"]}
                for r in all_results
            ],
            "mean": {k: v.tolist() for k, v in mean.items()},
            "std":  {k: v.tolist() for k, v in std.items()},
        }, f, indent=2)
    print(f"Wrote {out_json}")

    # Plot: single axes, all 4 bucket pairs with std bands
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    xs = np.arange(n_layers)
    for pair in PAIRS:
        m = mean[pair]
        s = std[pair]
        color = PAIR_COLORS[pair]
        ax.plot(xs, m, marker="o", markersize=3.5, linewidth=1.8,
                color=color, label=PAIR_LABELS[pair])
        ax.fill_between(xs, m - s, m + s, color=color, alpha=0.15)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.axhline(1, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Layer (L)")
    ax.set_ylabel("Recovery toward clean logit-diff")
    ax.set_title(
        f"Attention-pattern zero-ablation on Llama Guard 3 (AdvBench suffix, N={N} attacks)\n"
        f"s = adversarial positions (~20 toks), c = scaffold_tail (~54 toks).  "
        f"Shaded = ±1 std.",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = out_dir / "attn_pattern_recovery.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
