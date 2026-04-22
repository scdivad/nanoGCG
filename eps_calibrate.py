"""Measure GCG activation shifts at every layer for Llama Guard 3.

Analogous to harm_classifiers/eps_calibrate.py but for a decoder-only
causal LM (Llama Guard 3 8B, 32 layers, hidden_dim=4096) attacked via
examples/llama_guard.py (which saves per-prompt .pt files).

For each successful attack we do two forward passes — one on the clean
prompt ids and one on the attacked prompt ids — hook every transformer
layer, and compare activations. Because attention is causal and the two
inputs share a prefix up to where the suffix is inserted, the
activations are identical there; the interesting shift happens from
that divergence point onward (and at suffix positions, which exist
only in the attacked input).

Alignment (important — different from the BERT script):
  clean: [taxonomy prefix | user prompt | taxonomy end + <|eot|>...]
  adv:   [taxonomy prefix | user prompt | SUFFIX | taxonomy end + <|eot|>...]
Both share the first `div` positions. Adv has `n_s = n_adv - n_clean`
extra tokens in the middle. The SAME template-end tokens occupy
positions [div+n_s : n_adv) in adv and [div : n_clean) in clean, with
different activations because adv's past includes the suffix.

Usage:
    # Base-model attack results
    python eps_calibrate.py --pt-dir results/pt_<jobid>_base_i-gcg

    # LAT-model attack results (needs peft installed + adapter path)
    python eps_calibrate.py --pt-dir results/pt_<jobid>_lat_i-gcg \
        --adapter-path /path/to/checkpoint_1

    # Write raw per-attack stats for downstream plotting
    python eps_calibrate.py --pt-dir results/pt_<jobid>_base_i-gcg \
        --output-json results/eps_base.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_llama_layers(model):
    """Walk down PeftModel/base-model wrappers to the actual .layers list."""
    m = model
    # Follow a chain of wrapper attrs; on a plain AutoModelForCausalLM this
    # is model.model.layers; on a PeftModel it's base_model.model.model.layers.
    for _ in range(6):
        if hasattr(m, "layers"):
            return m.layers
        nxt = None
        for attr in ("model", "base_model", "module"):
            if hasattr(m, attr):
                nxt = getattr(m, attr)
                break
        if nxt is None:
            break
        m = nxt
    raise RuntimeError(f"Could not find .layers on {type(model).__name__}")


def capture_all(model, layers, input_ids):
    """Forward pass; return dict of key -> hidden_states (1, seq, dim).

    key == "embed" for the input-embeddings output and the integer
    layer index for each transformer block's output.
    """
    captured: Dict = {}

    def emb_hook(_m, _i, out):
        captured["embed"] = out.detach().clone()

    def make_layer_hook(idx):
        def h(_m, _i, out):
            # Llama decoder layer returns (hidden_states, ...) tuples.
            captured[idx] = (out[0] if isinstance(out, tuple) else out).detach().clone()
        return h

    hooks = [model.get_input_embeddings().register_forward_hook(emb_hook)]
    for idx, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_layer_hook(idx)))

    with torch.no_grad():
        model(input_ids=input_ids.unsqueeze(0).to(model.device))

    for h in hooks:
        h.remove()
    return captured


def find_divergence(a: torch.Tensor, b: torch.Tensor) -> int:
    """First position where 1-D token id sequences differ; len(shorter) if one is a prefix of the other."""
    n = min(a.shape[0], b.shape[0])
    for i in range(n):
        if a[i].item() != b[i].item():
            return i
    return n


def load_pt_records(pt_dir: Optional[str], pt_files: Optional[List[str]]) -> List[dict]:
    """Load records from a directory of .pt files or an explicit list."""
    paths: List[Path] = []
    if pt_dir:
        paths.extend(sorted(Path(pt_dir).glob("*.pt")))
    if pt_files:
        paths.extend(Path(p) for p in pt_files)
    if not paths:
        raise SystemExit("No .pt files found. Pass --pt-dir or --pt-file.")
    records = []
    for p in paths:
        rec = torch.load(p, map_location="cpu", weights_only=False)
        rec["_path"] = str(p)
        records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-dir", type=str, default=None,
                    help="Directory of prompt_NNN.pt files from examples/llama_guard.py")
    ap.add_argument("--pt-file", type=str, action="append", default=None,
                    help="Individual .pt file (repeatable). Mutually additive with --pt-dir.")
    ap.add_argument("--model", type=str,
                    default="/home/dcheung2/new/guard_lat/Llama-Guard-3-8B",
                    help="Path to Llama Guard 3 8B base weights.")
    ap.add_argument("--adapter-path", type=str, default=None,
                    help="Optional PEFT/LoRA adapter. Use the SAME adapter the "
                    "attacks were generated against (base vs LAT matters — running "
                    "LAT attacks through the base model gives meaningless activations).")
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--device-map", type=str, default="auto")
    ap.add_argument("--max-n", type=int, default=None,
                    help="Cap how many successful attacks to process (for quick tests).")
    ap.add_argument("--output-json", type=str, default=None,
                    help="Optional path to write raw per-attack per-layer stats.")
    args = ap.parse_args()

    print(f"[eps] loading .pt records")
    records = load_pt_records(args.pt_dir, args.pt_file)
    print(f"  found {len(records)} records")

    succ = [r for r in records if r.get("success") is True]
    print(f"  {len(succ)} successful attacks (pre=unsafe -> post=safe)")
    if args.max_n is not None:
        succ = succ[: args.max_n]
        print(f"  processing first {len(succ)} after --max-n cap")
    if not succ:
        raise SystemExit("No successful attacks in input. Re-run once you have flips.")

    print(f"[eps] loading model: {args.model}")
    # Note: tokenizer not strictly needed here (we use saved ids), but we load
    # it so optional decode is available and to mirror the attack-driver setup.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
        device_map=args.device_map,
    )
    model.eval()
    if args.adapter_path:
        from peft import PeftModel
        print(f"[eps] attaching adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model.eval()

    layers = get_llama_layers(model)
    num_layers = len(layers)
    print(f"[eps] found {num_layers} transformer layers")

    keys: List = ["embed"] + list(range(num_layers))
    stats = {k: {
        # Diff stats on ALIGNED shifted-content positions (same tokens, different
        # activations due to suffix in past).
        "l2_global": [],
        "l2_per_pos_max": [],
        "l2_per_pos_mean": [],
        "l2_per_pos_p90": [],
        "l2_per_pos_median": [],
        # Suffix positions — exist only in adv; we record the MAGNITUDE of adv
        # activations at those positions (no aligned clean activation exists).
        "suffix_l2_global": [],
        "suffix_l2_per_pos_max": [],
        # Clean-only per-position magnitudes for the lat_baseline_v2 convention
        # (pos_cap = p% of median clean per-pos L2).
        "clean_per_pos_max": [],
        "clean_per_pos_mean": [],
        "clean_per_pos_median": [],
    } for k in keys}

    raw_records: List[dict] = []  # only populated if --output-json

    n_used = 0
    for i, atk in enumerate(succ):
        prompt_ids = atk["prompt_ids"].to(torch.long)
        adv_ids = atk["attacked_prompt_ids"].to(torch.long)
        n_c = prompt_ids.shape[0]
        n_a = adv_ids.shape[0]
        n_s = n_a - n_c
        if n_s <= 0:
            # Shouldn't happen for a real suffix attack; skip defensively.
            print(f"  skip {atk.get('_path')}: n_adv<=n_clean (n_c={n_c}, n_a={n_a})")
            continue

        div = find_divergence(prompt_ids, adv_ids)
        if div >= n_c:
            # No difference inside clean's range — suspicious.
            print(f"  skip {atk.get('_path')}: divergence >= n_clean (div={div}, n_c={n_c})")
            continue

        clean_acts = capture_all(model, layers, prompt_ids)
        adv_acts = capture_all(model, layers, adv_ids)

        # Expected alignment for "same tokens, shifted past":
        #   adv_ids[div + n_s : n_a]  == prompt_ids[div : n_c]  (element-wise)
        # Check once on the first record to catch misalignment early.
        if n_used == 0:
            left = adv_ids[div + n_s : n_a]
            right = prompt_ids[div : n_c]
            if left.shape == right.shape and not torch.equal(left, right):
                print(f"  WARNING: post-suffix tokens don't match between clean and adv. "
                      f"Alignment may be off for this attack set — stats still computed "
                      f"but interpret with caution.")

        per_attack_stats: Dict = {}

        for k in keys:
            clean_h = clean_acts[k].squeeze(0)   # (n_c, d)
            adv_h = adv_acts[k].squeeze(0)       # (n_a, d)

            # Shifted content: same tokens, different past → activations differ.
            # clean positions [div:n_c) correspond to adv positions [div+n_s:n_a).
            content_clean = clean_h[div:n_c]
            content_adv_shifted = adv_h[div + n_s : n_a]
            if content_clean.shape != content_adv_shifted.shape:
                # Can happen only if there's odd tokenizer behavior; trim to min.
                m = min(content_clean.shape[0], content_adv_shifted.shape[0])
                content_clean = content_clean[:m]
                content_adv_shifted = content_adv_shifted[:m]

            diff = (content_clean - content_adv_shifted).float()
            l2_global = diff.norm(p=2).item()
            per_pos = diff.norm(p=2, dim=1)  # (n_content,)

            stats[k]["l2_global"].append(l2_global)
            stats[k]["l2_per_pos_max"].append(per_pos.max().item() if per_pos.numel() else 0.0)
            stats[k]["l2_per_pos_mean"].append(per_pos.mean().item() if per_pos.numel() else 0.0)
            stats[k]["l2_per_pos_p90"].append(
                torch.quantile(per_pos, 0.9).item() if per_pos.numel() else 0.0)
            stats[k]["l2_per_pos_median"].append(
                torch.quantile(per_pos, 0.5).item() if per_pos.numel() else 0.0)

            # Suffix-only magnitudes (adv positions [div : div+n_s))
            suffix_act = adv_h[div : div + n_s].float()
            s_pp = suffix_act.norm(p=2, dim=1)
            stats[k]["suffix_l2_global"].append(suffix_act.norm(p=2).item())
            stats[k]["suffix_l2_per_pos_max"].append(s_pp.max().item() if s_pp.numel() else 0.0)

            # Clean-only magnitudes for LAT eps conventions.
            clean_pp = clean_h.float().norm(p=2, dim=1)  # (n_c,)
            stats[k]["clean_per_pos_max"].append(clean_pp.max().item())
            stats[k]["clean_per_pos_mean"].append(clean_pp.mean().item())
            stats[k]["clean_per_pos_median"].append(
                torch.quantile(clean_pp, 0.5).item())

            if args.output_json is not None:
                per_attack_stats[str(k)] = {
                    "l2_global": l2_global,
                    "l2_per_pos_max": float(per_pos.max().item() if per_pos.numel() else 0.0),
                    "suffix_l2_global": float(suffix_act.norm(p=2).item()),
                }

        n_used += 1
        if args.output_json is not None:
            raw_records.append({
                "prompt": atk.get("prompt", ""),
                "n_clean": int(n_c),
                "n_adv": int(n_a),
                "n_suffix_tokens": int(n_s),
                "divergence": int(div),
                "per_layer": per_attack_stats,
            })

        if n_used % 10 == 0:
            print(f"  processed {n_used}/{len(succ)}")

    print(f"\n[eps] {n_used} attacks analyzed")

    # ==================== Print tables ====================
    tag = "base" if args.adapter_path is None else "LAT"
    print(f"\n{'=' * 90}")
    print(f"LLAMA GUARD ACTIVATION SHIFTS — {tag}")
    print(f"{'=' * 90}")
    print(f"  Model: {args.model}")
    print(f"  Adapter: {args.adapter_path or '<none>'}")
    print(f"  Records:  {n_used} successful attacks")
    print(f"  Layers:   {num_layers}")

    def label(k):
        return f"L{k:02d}" if isinstance(k, int) else "Embed"

    print(f"\n{'-' * 90}")
    print("PART 1: GLOBAL L2 (shifted content positions; same tokens / different past)")
    print(f"{'-' * 90}")
    print(f"  {'Layer':<8} {'Mean':>10} {'Med':>10} {'P75':>10} {'P90':>10} {'Max':>10}")
    print(f"  {'-' * 58}")
    for k in keys:
        v = stats[k]["l2_global"]
        print(f"  {label(k):<8} {np.mean(v):>10.2f} {np.median(v):>10.2f} "
              f"{np.percentile(v, 75):>10.2f} {np.percentile(v, 90):>10.2f} {np.max(v):>10.2f}")

    print(f"\n{'-' * 90}")
    print("PART 2: PER-POSITION L2 (mean across attacks, on shifted content)")
    print(f"{'-' * 90}")
    print(f"  {'Layer':<8} {'MaxPos':>10} {'MeanPos':>10} {'MedPos':>10} {'P90Pos':>10}")
    print(f"  {'-' * 50}")
    for k in keys:
        s = stats[k]
        print(f"  {label(k):<8} {np.mean(s['l2_per_pos_max']):>10.2f} "
              f"{np.mean(s['l2_per_pos_mean']):>10.2f} "
              f"{np.mean(s['l2_per_pos_median']):>10.2f} "
              f"{np.mean(s['l2_per_pos_p90']):>10.2f}")

    print(f"\n{'-' * 90}")
    print("PART 3: SUFFIX-ONLY ACTIVATION MAGNITUDE (adv positions; no clean counterpart)")
    print(f"{'-' * 90}")
    print(f"  {'Layer':<8} {'Sfx_Gbl_Mean':>14} {'Sfx_MaxPos_Mean':>18}")
    print(f"  {'-' * 42}")
    for k in keys:
        s = stats[k]
        print(f"  {label(k):<8} {np.mean(s['suffix_l2_global']):>14.2f} "
              f"{np.mean(s['suffix_l2_per_pos_max']):>18.2f}")

    # ==================== Suggested eps ranges ====================
    print(f"\n{'=' * 90}")
    print("SUGGESTED EPS RANGES FOR LAT SWEEP  (median / P75 of global L2 diff)")
    print(f"{'=' * 90}")
    print(f"  {'Layer':<8} {'Med_Gbl':>10} {'P75_Gbl':>10} {'Med_PosMax':>12} {'P75_PosMax':>12}")
    print(f"  {'-' * 58}")
    for k in keys:
        s = stats[k]
        med_g = np.median(s["l2_global"])
        p75_g = np.percentile(s["l2_global"], 75)
        med_pm = np.median(s["l2_per_pos_max"])
        p75_pm = np.percentile(s["l2_per_pos_max"], 75)
        print(f"  {label(k):<8} {med_g:>10.2f} {p75_g:>10.2f} {med_pm:>12.2f} {p75_pm:>12.2f}")

    print(f"\n{'=' * 90}")
    print("POS_CAP TIERS  (lat_baseline_v2 convention: % of median clean per-pos L2)")
    print(f"{'=' * 90}")
    print(f"  {'Layer':<8} {'CleanMed':>10} {'5pct':>8} {'10pct':>8} {'20pct':>8} {'50pct':>8}")
    print(f"  {'-' * 54}")
    for k in keys:
        s = stats[k]
        base = np.median(s["clean_per_pos_median"])
        print(f"  {label(k):<8} {base:>10.2f} {0.05 * base:>8.2f} {0.10 * base:>8.2f} "
              f"{0.20 * base:>8.2f} {0.50 * base:>8.2f}")

    if args.output_json:
        # Summary + per-attack raw records.
        summary = {k: {
            sk: [float(x) for x in v]
            for sk, v in stats[k].items()
        } for k in keys}
        # Keys in summary get stringified so JSON is valid.
        summary = {str(k): v for k, v in summary.items()}
        payload = {
            "model": args.model,
            "adapter_path": args.adapter_path,
            "num_layers": num_layers,
            "n_used": n_used,
            "summary": summary,
            "per_attack": raw_records,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[eps] raw stats -> {args.output_json}")


if __name__ == "__main__":
    main()
