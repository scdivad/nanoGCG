"""
Select long unsafe prompts from the Aegis dataset for activation-patching
experiments where prompt-content tokens >> adversarial-suffix tokens.

Pipeline:
  1. Load aegis, keep label > 0 (unsafe per Aegis's taxonomy).
  2. Keep texts whose Llama-Guard-3 tokenization length is in [MIN_LEN,
     MAX_LEN]. Lower bound makes content dominate the 20-tok suffix;
     upper bound caps per-attack forward-pass cost.
  3. Run Llama Guard classify() on each; keep only those it also flags as
     'unsafe' (Aegis labels don't 1:1 correspond to LG's policy).
  4. Write N of them (or all surviving) to JSONL with a 'prompt' key per
     line. The driver's --prompts-file accepts .jsonl and handles
     embedded newlines.

Outputs:
  - <out-jsonl>  (one {"prompt": "..."} per line)
"""
import argparse
import json
import random
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


def classify(model, tok, prompt: str, max_new_tokens: int = 20) -> str:
    input_ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt", add_generation_prompt=True,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aegis-path", default="/home/davidsc2/FOCAL/ctlm/pulled/harm_classifiers/datasets/aegis")
    ap.add_argument("--lg-path",    default="/home/davidsc2/FOCAL/ctlm/pulled/Llama-Guard-3-8B")
    ap.add_argument("--split",      default="train")
    ap.add_argument("--min-len",    type=int, default=100, help="min prompt-text tokens (want content >> 20-tok suffix)")
    ap.add_argument("--max-len",    type=int, default=300, help="max prompt-text tokens (attack cost grows roughly linearly)")
    ap.add_argument("--n-want",     type=int, default=15, help="target number of prompts to keep")
    ap.add_argument("--n-candidates", type=int, default=60, help="how many length-filtered candidates to feed through LG")
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--out-jsonl",  default="/home/davidsc2/FOCAL/ctlm/pulled/nanoGCG/aegis_prompts.jsonl")
    args = ap.parse_args()

    random.seed(args.seed)

    ds = load_from_disk(args.aegis_path)
    tok = AutoTokenizer.from_pretrained(args.lg_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.lg_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    subset = ds[args.split].filter(lambda x: x["label"] > 0)
    print(f"Aegis {args.split} unsafe (label>0): {len(subset)} rows")

    # Length-filter.
    texts = subset["text"]
    with_lens = []
    for t in texts:
        L = len(tok(t, add_special_tokens=False).input_ids)
        if args.min_len <= L <= args.max_len:
            with_lens.append((t, L))
    print(f"  after length filter [{args.min_len}, {args.max_len}]: {len(with_lens)}")

    random.shuffle(with_lens)
    pool = with_lens[: args.n_candidates]
    print(f"  sampling {len(pool)} candidates through Llama Guard...")

    kept = []
    for i, (t, L) in enumerate(pool, start=1):
        label = classify(model, tok, t)
        first = label.split("\n", 1)[0].strip().lower()
        keep = first.startswith("unsafe")
        tag = "✓" if keep else "✗"
        print(f"  [{i}/{len(pool)}] {tag} LG={first!r:14s} len={L:4d}  {t[:80]!r}")
        if keep:
            kept.append(t)
        if len(kept) >= args.n_want:
            break

    print(f"\nKept {len(kept)}/{args.n_want} prompts that LG agrees are unsafe.")
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for t in kept:
            f.write(json.dumps({"prompt": t}) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
