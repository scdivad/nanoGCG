"""
Overlay 'clean' vs 'zero' patching recovery curves.

Clean patch: replace module output at suffix positions with the length-matched
             benign '!'-filler clean's activation (what the model would have
             done at those positions given a benign input).
Zero patch:  replace module output at suffix positions with 0 (pure ablation).

Both interventions measure "how much of the attack signal flows through this
component at these positions?" But they measure different things:
  - clean patch: "how much does the ADVERSARIAL content at this component
                 contribute, over and above what a benign input would have
                 produced?"
  - zero patch:  "how much does the module's contribution along this pathway
                 matter at all at these positions?"

When clean > zero, the adversarial activation differs from benign in a way
that preserves magnitude — zeroing it out is more destructive than
replacing with clean.
When zero > clean, benign activation is non-trivial too; zeroing removes
both benign signal and adversarial signal.

We focus on the suffix position-set since that's where the attack lives.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = ["q_proj", "k_proj", "v_proj", "o_proj", "mlp"]
COMPONENT_COLORS = {
    "q_proj": "#1f77b4",
    "k_proj": "#ff7f0e",
    "v_proj": "#2ca02c",
    "o_proj": "#d62728",
    "mlp":    "#9467bd",
}


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-json", required=True)
    ap.add_argument("--zero-json",  required=True)
    ap.add_argument("--out-png",    required=True)
    ap.add_argument("--pset",       default="suffix", choices=["suffix", "content", "last"])
    args = ap.parse_args()

    c = load(args.clean_json)
    z = load(args.zero_json)
    n_layers = c["n_layers"]
    assert z["n_layers"] == n_layers
    xs = np.arange(n_layers)

    # One subplot per component, overlay clean/zero mean ± std.
    fig, axes = plt.subplots(1, len(COMPONENTS), figsize=(22, 4.2), sharey=True)
    for ax, comp in zip(axes, COMPONENTS):
        for mode, data, dash in [("clean", c, "-"), ("zero", z, "--")]:
            m = np.array(data["mean"][comp][args.pset])
            s = np.array(data["std"][comp][args.pset])
            color = COMPONENT_COLORS[comp]
            ax.plot(xs, m, linestyle=dash, color=color, marker="o", markersize=3.5,
                    linewidth=1.8, label=f"{mode}")
            ax.fill_between(xs, m - s, m + s, color=color, alpha=0.12)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.axhline(1, color="gray", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Layer (L)")
        ax.set_title(comp)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("Recovery toward clean logit-diff")

    N = c["n_prompts"]
    fig.suptitle(
        f"Clean-patch (solid) vs. Zero-patch (dashed) — patching at '{args.pset}' positions, N={N} attacks\n"
        f"Positive = corrupted logit-diff moves toward clean; Negative = moves further from clean (worse than attacked).",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
