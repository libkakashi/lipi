"""
LoRA Adapter Injection for Lipi Encoder.

Injects LoRA into Stage 2 and Stage 3 attention blocks.
Stage 1 remains completely frozen (learns script-universal features).

Uses HuggingFace PEFT for LoRA management, which integrates
with Microsoft Olive for .onnx_adapter export.

IMPORTANT: MLP submodules use named attributes (fc1, fc2), not
Sequential indexing. The target_modules list must match the actual
attribute names in the encoder's module tree.
"""

import torch.nn as nn
from peft import get_peft_model, LoraConfig


def get_lora_target_modules(
    stage2_blocks: int = 4,
    stage3_blocks: int = 3,
) -> list[str]:
    """Build the list of module paths that should get LoRA adapters.

    Targets QKV, output projection, and MLP layers in Stage 2 and Stage 3.
    Stage 1 is excluded (frozen, script-universal features).

    Returns:
        List of module path strings matching the encoder's named_modules.
    """
    targets = []

    for i in range(stage2_blocks):
        targets.extend([
            f"stage2.{i}.attn.qkv",
            f"stage2.{i}.attn.proj",
            f"stage2.{i}.mlp.fc1",
            f"stage2.{i}.mlp.fc2",
        ])

    for i in range(stage3_blocks):
        targets.extend([
            f"stage3.{i}.attn.qkv",
            f"stage3.{i}.attn.proj",
            f"stage3.{i}.mlp.fc1",
            f"stage3.{i}.mlp.fc2",
        ])

    return targets


def inject_lora(
    backbone: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    stage2_blocks: int = 4,
    stage3_blocks: int = 3,
) -> nn.Module:
    """Inject LoRA adapters into Stage 2 and Stage 3 of the encoder.

    Stage 1 remains completely frozen. Only LoRA parameters are trainable.

    Args:
        backbone: LipiEncoder instance.
        rank: LoRA rank (default 16).
        alpha: LoRA alpha scaling (default 32).
        dropout: LoRA dropout (default 0.05).
        stage2_blocks: Number of blocks in stage 2.
        stage3_blocks: Number of blocks in stage 3.

    Returns:
        PEFT-wrapped model with LoRA adapters injected.
    """
    target_modules = get_lora_target_modules(stage2_blocks, stage3_blocks)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
    )

    model = get_peft_model(backbone, lora_config)

    # Verify: only LoRA params should be trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return model


def get_lora_param_count(
    rank: int = 16,
    stage2_dim: int = 384,
    stage2_blocks: int = 4,
    stage3_dim: int = 384,
    stage3_blocks: int = 3,
    stage2_mlp_ratio: int = 3,
    stage3_mlp_ratio: int = 4,
) -> dict[str, int]:
    """Calculate expected LoRA parameter count without instantiating.

    Returns breakdown by component.
    """
    counts = {}

    # Stage 2: per block
    # qkv: (dim, 3*dim) -> LoRA: dim*rank + rank*3*dim = rank*(dim + 3*dim) = rank*4*dim
    # proj: (dim, dim) -> LoRA: dim*rank + rank*dim = 2*rank*dim
    # fc1: (dim, dim*mlp_ratio) -> LoRA: dim*rank + rank*dim*mlp_ratio
    # fc2: (dim*mlp_ratio, dim) -> LoRA: dim*mlp_ratio*rank + rank*dim
    s2_per_block = (
        rank * (stage2_dim + 3 * stage2_dim)  # qkv
        + rank * 2 * stage2_dim               # proj
        + rank * (stage2_dim + stage2_dim * stage2_mlp_ratio)  # fc1
        + rank * (stage2_dim * stage2_mlp_ratio + stage2_dim)  # fc2
    )
    counts["stage2"] = s2_per_block * stage2_blocks

    # Stage 3: same pattern
    s3_per_block = (
        rank * (stage3_dim + 3 * stage3_dim)
        + rank * 2 * stage3_dim
        + rank * (stage3_dim + stage3_dim * stage3_mlp_ratio)
        + rank * (stage3_dim * stage3_mlp_ratio + stage3_dim)
    )
    counts["stage3"] = s3_per_block * stage3_blocks

    counts["total"] = counts["stage2"] + counts["stage3"]
    return counts
