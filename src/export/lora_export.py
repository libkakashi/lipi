"""
LoRA Adapter Export to ONNX Runtime format.

Converts HuggingFace PEFT LoRA adapters to .onnx_adapter format
for runtime adapter swapping via ORT MultiLoRA API.

Uses onnxruntime's native AdapterFormat when available,
falls back to Microsoft Olive toolchain otherwise.
"""

import json
from pathlib import Path

import torch
import numpy as np


def export_adapter_native(
    lora_state_dict: dict,
    output_path: str | Path,
):
    """Export LoRA weights using ONNX Runtime's native AdapterFormat.

    This is the preferred method when onnxruntime >= 1.24.0 is available.

    Args:
        lora_state_dict: Dict of LoRA parameter names -> tensors.
            Keys should match the parameter names in the ONNX model.
        output_path: Output .onnx_adapter file path.
    """
    try:
        import onnxruntime as ort
        adapter_format = ort.AdapterFormat()
    except (ImportError, AttributeError):
        raise RuntimeError(
            "onnxruntime >= 1.24.0 required for native adapter export. "
            "Install with: pip install onnxruntime>=1.24.0"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert PyTorch tensors to OrtValues
    params = {}
    for name, tensor in lora_state_dict.items():
        arr = tensor.detach().cpu().numpy().astype(np.float16)
        params[name] = ort.OrtValue.ortvalue_from_numpy(arr)

    adapter_format.set_parameters(params)
    adapter_format.export_adapter(str(output_path))

    size_mb = output_path.stat().st_size / 1e6
    print(f"Exported adapter to: {output_path} ({size_mb:.2f} MB, {len(params)} params)")


def export_adapter_olive(
    base_model_path: str | Path,
    adapter_path: str | Path,
    output_path: str | Path,
    language: str,
):
    """Export LoRA adapter using Microsoft Olive toolchain.

    Fallback method when native ORT export isn't available.

    Args:
        base_model_path: Path to the base ONNX model.
        adapter_path: Path to PEFT adapter directory.
        output_path: Output .onnx_adapter file path.
        language: Language code for logging.
    """
    import subprocess

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Olive convert-adapters
    cmd = [
        "olive", "convert-adapters",
        "--adapter_path", str(adapter_path),
        "--output_path", str(output_path),
        "--dtype", "float16",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Olive export failed:\n{result.stderr}")
        raise RuntimeError(f"Olive adapter export failed for {language}")

    print(f"Exported {language} adapter to: {output_path}")


def extract_lora_state_dict(checkpoint_path: str | Path) -> dict:
    """Extract LoRA-only parameters from a Phase 2 checkpoint.

    Args:
        checkpoint_path: Path to adapter training checkpoint.

    Returns:
        Dict of LoRA parameter names -> tensors.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if "lora" in checkpoint:
        return checkpoint["lora"]

    # Extract from full model state dict
    lora_params = {
        k: v for k, v in checkpoint.items()
        if "lora_" in k
    }

    if not lora_params:
        raise ValueError(f"No LoRA parameters found in {checkpoint_path}")

    return lora_params
