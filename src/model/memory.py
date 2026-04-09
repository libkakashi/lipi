"""VRAM budget estimation for LipiMoEEncoder.

Estimates the maximum pixel budget (B*W) that fits in GPU memory,
accounting for model params, optimizer state, and activation memory
at each stage of the encoder.
"""

def estimate_pixel_budget(model, vram_gb: float = 32,
                          margin: float = 0.50) -> int:
    """Estimate max pixel budget (B*W) from model architecture and VRAM.

    Memory accounting per component:
    - Stem (not checkpointed): conv intermediates
    - Shared SWA (entire block checkpointed, dim/2): 1 tensor per block
    - Expert pools: height pool + width pool intermediates
    - Expert SWA blocks (inner attn/MLP checkpointed, dim): ~5 tensors per block
    - Recompute peak: one block recomputes forward during backward

    Spatial resolutions (per input width W):
        Stem/Shared: h=16, w=W/2  → tokens_per_W = 8
        After pool1:  h=8,  w=W/4  → tokens_per_W = 2
        After pool2:  h=4,  w=W/8  → tokens_per_W = 0.5
    """
    shared_dim = model.shared_swa[0].norm1.normalized_shape[0]
    n_shared = len(model.shared_swa)
    shared_mlp_ratio = model.shared_swa[0].mlp.fc1.out_features // shared_dim
    dim = model.stage1[0].norm1.normalized_shape[0]
    n_stage1 = len(model.stage1)
    stage1_mlp_ratio = model.stage1[0].expert_mlps[0].fc1.out_features // dim
    n_stage2 = len(model.stage2)
    stage2_mlp_ratio = model.stage2[0].expert_mlps[0].fc1.out_features // dim
    stem_ch = model.stem.layers[0].out_channels
    max_vocab = max(m.max_vocab for m in model.ctc_modules)
    enc_out_dim = model.enc_out_dim

    elems = 0

    # Stem (not checkpointed): ~3 conv layers worth of intermediates
    # Runs at h=16, w=W/2 → tokens_per_W = 8 (but in conv channels, not dim)
    elems += 3 * stem_ch * 16 * 0.5

    # Shared SWA: entire block checkpointed → saves only the input (1 tensor)
    # Runs at h=16, w=W/2 → tokens_per_W = 8, at shared_dim (dim/2)
    elems += n_shared * 1 * 8 * shared_dim

    # Expert Pool 1: height pool + width pool intermediates (at shared_dim)
    # Input at h=16, w=W/2 (tokens_per_W=8), output at h=8, w=W/4 (tokens_per_W=2)
    elems += 14 * shared_dim

    # Expert SWA Stage 1: ~5 tensors per block (not fully checkpointed)
    # Runs at h=8, w=W/4 → tokens_per_W = 2, at full dim
    elems += n_stage1 * 5 * 2 * dim

    # Expert Pool 2: same accounting as pool 1 but at smaller resolution and full dim
    # Input at h=8, w=W/4 (tokens_per_W=2), output at h=4, w=W/8 (tokens_per_W=0.5)
    elems += 4.5 * dim

    # Expert SWA Stage 2: ~5 tensors per block
    # Runs at h=4, w=W/8 → tokens_per_W = 0.5
    elems += n_stage2 * 5 * 0.5 * dim

    # CTC logits + fold output at tokens_per_W = 0.125
    elems += max_vocab * 0.125 + enc_out_dim * 0.125

    # Recompute peak: during backward, one block recomputes its full forward,
    # temporarily holding norm + QKV + attn_out + MLP intermediates.
    # ~(7 + 2*mlp_ratio) * dim per token for the largest block.
    def _recompute_peak(d, mlp_ratio, tokens_per_px):
        return (7 + 2 * mlp_ratio) * d * tokens_per_px

    recompute_peak = max(
        _recompute_peak(shared_dim, shared_mlp_ratio, 8),
        _recompute_peak(dim, stage1_mlp_ratio, 2),
        _recompute_peak(dim, stage2_mlp_ratio, 0.5),
    )

    # bf16 = 2 bytes/element
    bytes_per_pixel_col = int((elems + recompute_peak) * 2)

    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    fixed = model_bytes * 4  # params + grads + adam m + adam v
    available = vram_gb * 1e9 * margin - fixed
    pixel_budget = int(available / bytes_per_pixel_col)

    print(f"  VRAM estimate: {model_bytes/1e9:.2f}GB model, "
          f"{fixed/1e9:.2f}GB fixed, "
          f"{bytes_per_pixel_col/1e3:.0f}KB/px, "
          f"{available/1e9:.1f}GB for activations")
    print(f"  Pixel budget: {pixel_budget}")

    return pixel_budget
