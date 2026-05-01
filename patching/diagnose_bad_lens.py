"""
Diagnose bad_lens in trajectory_attribution_topk.py: why does ACG produce
trajectory states with suffix length != final attacked suffix length?

For each .pt in --pt-dir (or --limit prompts), reports:
  * len(attacked_prompt_ids)
  * len(prompt_ids)
  * suffix_len from align_suffix_positions
  * len(all_suffix_ids[t]) for each t — flag mismatches
  * For the first few mismatched (prompt, t) pairs, decodes both the saved
    final suffix region and the per-step suffix to show what differs.

Usage:
  python patching/diagnose_bad_lens.py \\
      --pt-dir results/pt_48670390_lg1_base_acg \\
      --tokenizer /home/dcheung2/new/guard_lat/LlamaGuard-7b \\
      --limit 10
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_sweep import align_suffix_positions


def to_long_1d(x):
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
    else:
        t = torch.as_tensor(x)
    return t.long().squeeze()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-dir", required=True)
    ap.add_argument("--tokenizer", required=True,
                    help="HF tokenizer path/name for decoding suffixes.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--show-mismatch", type=int, default=5,
                    help="How many mismatched (prompt, step) pairs to "
                    "print in detail (token IDs + decoded text).")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    pt_files = sorted(Path(args.pt_dir).glob("prompt_*.pt"))
    if args.limit:
        pt_files = pt_files[: args.limit]
    print(f"Inspecting {len(pt_files)} .pt files from {args.pt_dir}\n")

    summary = {
        "n_total":         0,
        "n_perfect":       0,
        "n_with_mismatch": 0,
        "len_diffs":       Counter(),
        "first_step_mismatched": Counter(),  # at which step does the first mismatch appear
    }
    shown = 0

    for p in pt_files:
        d = torch.load(p, weights_only=False)
        if "all_suffix_ids" not in d:
            continue
        attacked = to_long_1d(d["attacked_prompt_ids"])
        prompt   = to_long_1d(d["prompt_ids"])
        s_start, s_end = align_suffix_positions(prompt, attacked)
        suffix_len = s_end - s_start

        all_suff = d["all_suffix_ids"]
        if not isinstance(all_suff, (list, tuple)):
            all_suff = [all_suff]

        per_step_lens = [int(to_long_1d(s).numel()) for s in all_suff]
        len_counts = Counter(per_step_lens)
        most_common_len, mc_count = len_counts.most_common(1)[0]

        n_match  = sum(1 for l in per_step_lens if l == suffix_len)
        n_mis    = len(per_step_lens) - n_match
        succeeded = bool(d.get("success"))
        post = d.get("post_verdict")

        print(f"{p.name}: success={succeeded} post={post}  "
              f"|attacked|={len(attacked)}  |prompt|={len(prompt)}  "
              f"final_suffix_len={suffix_len}  "
              f"per_step_lens(min/median/max)={min(per_step_lens)}/"
              f"{sorted(per_step_lens)[len(per_step_lens)//2]}/{max(per_step_lens)}  "
              f"steps_matching_final={n_match}/{len(per_step_lens)}")

        summary["n_total"] += 1
        if n_mis == 0:
            summary["n_perfect"] += 1
        else:
            summary["n_with_mismatch"] += 1
            for l in per_step_lens:
                summary["len_diffs"][l - suffix_len] += 1
            # First-mismatch step
            for t, l in enumerate(per_step_lens):
                if l != suffix_len:
                    summary["first_step_mismatched"][t] += 1
                    break

        # Decode and show details for the first few mismatched pairs
        if n_mis > 0 and shown < args.show_mismatch:
            # Show: saved final suffix tokens, vs first mismatched step's suffix
            saved_final = attacked[s_start:s_end]
            print(f"  saved final suffix    ({len(saved_final)} toks):")
            print(f"    ids:  {saved_final.tolist()}")
            print(f"    text: {tok.decode(saved_final, clean_up_tokenization_spaces=False)!r}")
            for t, suf in enumerate(all_suff):
                suf1d = to_long_1d(suf)
                if suf1d.numel() != suffix_len:
                    print(f"  all_suffix_ids[{t}] ({suf1d.numel()} toks, "
                          f"diff={suf1d.numel() - suffix_len:+d}):")
                    print(f"    ids:  {suf1d.tolist()}")
                    print(f"    text: {tok.decode(suf1d, clean_up_tokenization_spaces=False)!r}")
                    break
            # Also show the FINAL trajectory step for comparison
            last_suf = to_long_1d(all_suff[-1])
            if last_suf.numel() != saved_final.numel():
                print(f"  all_suffix_ids[-1]  ({last_suf.numel()} toks, "
                      f"diff_vs_saved={last_suf.numel() - saved_final.numel():+d}):")
                print(f"    ids:  {last_suf.tolist()}")
                print(f"    text: {tok.decode(last_suf, clean_up_tokenization_spaces=False)!r}")
            shown += 1
            print()

    print("\n========== SUMMARY ==========")
    print(f"  prompts inspected:        {summary['n_total']}")
    print(f"  zero-mismatch prompts:    {summary['n_perfect']}")
    print(f"  with-mismatch prompts:    {summary['n_with_mismatch']}")
    if summary["len_diffs"]:
        print(f"\n  Length-difference distribution (per_step_len - suffix_len):")
        for diff, cnt in sorted(summary["len_diffs"].items()):
            print(f"    diff={diff:+d}  count={cnt}")
        print(f"\n  First-mismatch-step distribution:")
        for t, cnt in sorted(summary["first_step_mismatched"].items()):
            print(f"    step={t:>3}  count={cnt}")


if __name__ == "__main__":
    main()
