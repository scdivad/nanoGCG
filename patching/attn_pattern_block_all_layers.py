"""
Block a given attention bucket pair at EVERY layer simultaneously, and
measure cumulative logit-diff recovery.

The layer-wise sweep tells you how much each layer's c->s attention
contributes on its own. Blocking c->s at all layers tests whether the
effect compounds — if tail positions can't attend to adversarial keys
anywhere in the stack, how much of the attack breaks?
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attn_pattern_patch import (
    compute_positions, install_attn_mask_hook, logit_diff,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-dir", required=True)
    ap.add_argument("--bucket", default="c->s", choices=["c->s", "s->c", "c->c", "s->s"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--include-last-in-c", action="store_true",
                    help="Append the final (verdict) position to the c bucket. "
                    "Makes c->s include the verdict position attending back to "
                    "adversarial keys, which is where the safe/unsafe decision "
                    "is committed.")
    ap.add_argument("--c-all-except-s", action="store_true",
                    help="Diagnostic: set c = every position except s. Blocks "
                    "ALL positions (header, user_content, tail, last) from "
                    "attending to adversarial keys. Most of these are already "
                    "causally masked; only tail and last are effective. "
                    "Recovery should equal the --include-last-in-c case.")
    ap.add_argument("--c-last-only", action="store_true",
                    help="Set c = {last position} only. Blocks the DIRECT "
                    "last -> suffix attention pathway; tail can still relay "
                    "adversarial content via tail -> suffix -> last. "
                    "Recovery estimates the direct-path contribution.")
    ap.add_argument("--tag", default="",
                    help="Optional slug appended to the output JSON filename "
                    "(e.g. --tag=advbench or --tag=aegis) so repeated runs on "
                    "different datasets don't overwrite each other.")
    ap.add_argument("--layers", default="all",
                    help="Comma-separated list of layer indices to enable the "
                    "block at (e.g. --layers 12,13). Default 'all' = every "
                    "layer.")
    args = ap.parse_args()

    pt_files = sorted(Path(args.pt_dir).glob("prompt_*.pt"))
    records = []
    for p in pt_files:
        d = torch.load(p, weights_only=False)
        if d.get("post_verdict") == "safe" and d.get("success"):
            records.append((p, d))
    if args.limit:
        records = records[: args.limit]
    print(f"Using {len(records)} successful attacks")

    tok_path = args.model_path or records[0][1]["tokenizer_name_or_path"]
    print(f"Loading model (eager attn): {tok_path}")
    tok = AutoTokenizer.from_pretrained(tok_path)
    model = AutoModelForCausalLM.from_pretrained(
        tok_path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    dev = model.device
    n_layers = model.config.num_hidden_layers

    NEWLINE_TOK = tok("\n\n",   add_special_tokens=False).input_ids[0]
    SAFE_TOK    = tok("safe",   add_special_tokens=False).input_ids[0]
    UNSAFE_TOK  = tok("unsafe", add_special_tokens=False).input_ids[0]
    FILLER_TOK  = tok("!",      add_special_tokens=False).input_ids[0]

    install_attn_mask_hook(model)

    if args.layers.lower() == "all":
        target_layers = list(range(n_layers))
    else:
        target_layers = [int(s) for s in args.layers.split(",") if s.strip()]
    print(f"Blocking at layers: {target_layers}")

    per_prompt = []
    recoveries = []
    for i, (p, d) in enumerate(records, start=1):
        atk_raw = d["attacked_prompt_ids"]
        clean_raw = d["prompt_ids"]
        attacked_ids = torch.cat([atk_raw, torch.tensor([NEWLINE_TOK])]).unsqueeze(0).to(dev)
        total_len = attacked_ids.shape[1]
        s_positions, c_positions = compute_positions(tok, clean_raw, atk_raw, total_len)
        if args.c_last_only:
            # Override c with ONLY the last position.
            c_positions = [total_len - 1]
        elif args.include_last_in_c:
            # Append the final position (verdict / appended '\n\n') to c
            last_pos = total_len - 1
            if last_pos not in c_positions:
                c_positions = c_positions + [last_pos]
        if args.c_all_except_s:
            # Override c with EVERY position not in s
            s_set = set(s_positions)
            c_positions = [p for p in range(total_len) if p not in s_set]

        # benign-filler clean
        benign_raw = atk_raw.clone()
        benign_raw[s_positions[0]:s_positions[-1] + 1] = FILLER_TOK
        benign_ids = torch.cat([benign_raw, torch.tensor([NEWLINE_TOK])]).unsqueeze(0).to(dev)

        # Baselines
        with torch.no_grad():
            clean_logits = model(benign_ids).logits
            corr_logits  = model(attacked_ids).logits
        clean_diff = logit_diff(clean_logits, SAFE_TOK, UNSAFE_TOK)
        corr_diff  = logit_diff(corr_logits,  SAFE_TOK, UNSAFE_TOK)
        gap = corr_diff - clean_diff
        if abs(gap) < 1e-6:
            continue

        # Look up which bucket to block.
        pair_positions = {
            "c->s": (c_positions, s_positions),
            "s->c": (s_positions, c_positions),
            "c->c": (c_positions, c_positions),
            "s->s": (s_positions, s_positions),
        }
        q_pos, k_pos = pair_positions[args.bucket]

        # Turn on the block at target layers
        for L in target_layers:
            model.model.layers[L].self_attn._custom_attn_mask_spec = (q_pos, k_pos)
        try:
            with torch.no_grad():
                p_logits = model(attacked_ids).logits
            p_diff = logit_diff(p_logits, SAFE_TOK, UNSAFE_TOK)
        finally:
            for L in target_layers:
                model.model.layers[L].self_attn._custom_attn_mask_spec = None

        rec = (corr_diff - p_diff) / gap
        recoveries.append(rec)
        print(f"[{i}/{len(records)}] {p.name}  "
              f"clean={clean_diff:+.3f}  attacked={corr_diff:+.3f}  patched={p_diff:+.3f}  "
              f"gap={gap:+.3f}  recovery={rec:+.2%}")
        per_prompt.append({
            "source_file": p.name,
            "prompt": d["prompt"],
            "clean_diff": clean_diff, "corr_diff": corr_diff, "patched_diff": p_diff,
            "gap": gap, "recovery": rec,
        })

    arr = np.array(recoveries)
    print()
    layers_desc = f"ALL {n_layers}" if args.layers.lower() == "all" else f"layers {target_layers}"
    print(f"=== block {args.bucket} at {layers_desc}, N={len(arr)} attacks ===")
    print(f"mean recovery = {arr.mean():+.2%}")
    print(f"  std         = {arr.std():+.2%}")
    print(f"  min / max   = {arr.min():+.2%} / {arr.max():+.2%}")

    out = {
        "bucket": args.bucket,
        "n_layers_blocked": n_layers,
        "n_prompts": len(arr),
        "mean_recovery": float(arr.mean()),
        "std_recovery": float(arr.std()),
        "min_recovery": float(arr.min()),
        "max_recovery": float(arr.max()),
        "per_prompt": per_prompt,
    }
    bucket_slug = args.bucket.replace("->", "_")
    suffix = []
    if args.c_last_only:       suffix.append("lastonly")
    if args.include_last_in_c: suffix.append("withlast")
    if args.c_all_except_s:    suffix.append("allexcepts")
    if args.tag:               suffix.append(args.tag)
    suffix_slug = ("_" + "_".join(suffix)) if suffix else ""
    out_path = Path(args.pt_dir).parent / f"block_all_layers_{bucket_slug}{suffix_slug}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
