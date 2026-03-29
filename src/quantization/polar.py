"""
PolarQuant Rotation for Transformer Weights.

Applies orthogonal rotation to QKV and MLP weight matrices
to spread outlier weights across dimensions. This improves
NVFP4 quantization fidelity by reducing the dynamic range
that needs to be represented in 4 bits.

Applied to Transformer weights ONLY (not GRU).

Reference: PolarQuant (2024) - Quantization-friendly weight rotation.
"""

import torch
import torch.nn as nn
from torch import Tensor


def compute_rotation_matrix(weight: Tensor) -> Tensor:
    """Compute optimal rotation matrix for a weight tensor.

    Computes a random orthogonal rotation of the input space that
    spreads outlier weight values across dimensions, improving
    quantization fidelity.

    For PolarQuant, we use a Hadamard-like random orthogonal matrix
    rather than SVD (which changes the weight shape for non-square matrices).

    Args:
        weight: (out_features, in_features) weight matrix.

    Returns:
        (in_features, in_features) orthogonal rotation matrix.
    """
    in_features = weight.shape[1]
    # Random orthogonal matrix via QR decomposition of random Gaussian
    random_matrix = torch.randn(in_features, in_features, device=weight.device, dtype=weight.dtype)
    Q, _ = torch.linalg.qr(random_matrix)
    return Q  # (in_features, in_features) orthogonal


def apply_polar_rotation(
    model: nn.Module,
    target_modules: list[str] | None = None,
) -> tuple[nn.Module, dict[str, Tensor]]:
    """Apply PolarQuant rotation to Transformer weight matrices.

    Args:
        model: The model to rotate.
        target_modules: List of module path prefixes to rotate.
            Defaults to QKV and MLP layers in all stages.

    Returns:
        (rotated_model, rotation_matrices) — the rotation matrices
        are needed for inference to un-rotate activations.
    """
    if target_modules is None:
        target_modules = ["stage1", "stage2", "stage3"]

    rotations = {}

    for name, module in model.named_modules():
        # Only rotate Linear layers in target stages
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.startswith(prefix) for prefix in target_modules):
            continue
        # Skip GRU-related modules
        if "gru" in name or "prediction_net" in name:
            continue

        weight = module.weight.data  # (out, in)
        if weight.ndim != 2:
            continue

        # Compute and apply rotation
        R = compute_rotation_matrix(weight)
        rotated_weight = weight @ R.T  # Rotate columns
        module.weight.data = rotated_weight
        rotations[name] = R

    print(f"Applied PolarQuant rotation to {len(rotations)} modules")
    return model, rotations


def measure_outlier_reduction(
    original_weight: Tensor, rotated_weight: Tensor
) -> dict:
    """Measure how much rotation reduced weight outliers.

    Returns metrics showing the improvement.
    """
    orig_range = original_weight.abs().max().item()
    rot_range = rotated_weight.abs().max().item()

    orig_std = original_weight.std().item()
    rot_std = rotated_weight.std().item()

    # Kurtosis (excess) — lower is better for quantization
    orig_kurtosis = (((original_weight - original_weight.mean()) / original_weight.std()) ** 4).mean().item() - 3
    rot_kurtosis = (((rotated_weight - rotated_weight.mean()) / rotated_weight.std()) ** 4).mean().item() - 3

    return {
        "range_reduction": 1 - rot_range / orig_range,
        "kurtosis_reduction": 1 - rot_kurtosis / max(orig_kurtosis, 1e-6),
        "original_range": orig_range,
        "rotated_range": rot_range,
        "original_kurtosis": orig_kurtosis,
        "rotated_kurtosis": rot_kurtosis,
    }
