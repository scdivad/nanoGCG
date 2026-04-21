config = GCGConfig(
    num_steps=500,          # Same ceiling as GCG/ACG; early_stop cuts it short
    search_width=64,        # ACG's "smaller batch size per iteration" — GCG default was 512
    topk=64,                # Tighter candidate pool, again for speed
    n_replace=4,            # Multi-position token swapping — ACG's key algorithmic win
    buffer_size=16,         # Historical attack buffer — ACG's other key algorithmic win
    early_stop=True,        # ACG's "cheap stopping condition"
    use_prefix_cache=True,  # Default on; big per-iteration speedup
    seed=42,
)