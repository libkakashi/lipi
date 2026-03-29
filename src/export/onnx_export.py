"""
ONNX Export for all Lipi model components.

Exports encoder, prediction net, joint net, and LID as separate ONNX files.
Uses opset 18 with dynamic_shapes for variable-width inputs.
"""

import torch
from pathlib import Path

from src.model.encoder import LipiEncoder
from src.model.prediction_net import PredictionNetwork
from src.model.joint_net import JointNetwork
from src.model.lid import MicroLID


def export_backbone(
    model_path: str | Path,
    output_path: str | Path,
    opset_version: int = 18,
):
    """Export the frozen backbone (without LoRA) to ONNX.

    Args:
        model_path: Path to Phase 1 checkpoint or encoder state dict.
        output_path: Output .onnx file path.
        opset_version: ONNX opset version.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = LipiEncoder()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if "encoder" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder"])
    else:
        encoder.load_state_dict(checkpoint)
    encoder.eval()

    dummy = torch.randn(1, 3, 32, 128)
    batch_dim = torch.export.Dim("batch", min=1, max=64)
    width_dim = torch.export.Dim("width", min=32, max=640)

    torch.onnx.export(
        encoder,
        (dummy,),
        str(output_path),
        input_names=["image"],
        output_names=["features", "lengths"],
        dynamic_shapes={"x": {0: batch_dim, 3: width_dim}},
        opset_version=opset_version,
    )

    print(f"Exported backbone to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")


def export_rnnt_head(
    pred_net_state: dict | str | Path,
    joint_net_state: dict | str | Path,
    output_dir: str | Path,
    language: str,
    vocab_size: int = 401,
    opset_version: int = 18,
):
    """Export Prediction Network and Joint Network as separate ONNX models.

    Args:
        pred_net_state: State dict or path to checkpoint.
        joint_net_state: State dict or path to checkpoint.
        output_dir: Directory for output files.
        language: Language code (e.g., 'hi', 'ta').
        vocab_size: Vocabulary size.
        opset_version: ONNX opset version.
    """
    output_dir = Path(output_dir) / language
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prediction Network
    pred_net = PredictionNetwork(vocab_size=vocab_size)
    if isinstance(pred_net_state, (str, Path)):
        state = torch.load(pred_net_state, map_location="cpu", weights_only=True)
        pred_net.load_state_dict(state.get("pred_net", state))
    else:
        pred_net.load_state_dict(pred_net_state)
    pred_net.eval()

    dummy_token = torch.tensor([[1]], dtype=torch.long)
    dummy_hidden = torch.zeros(1, 1, pred_net.hidden_dim)

    torch.onnx.export(
        pred_net,
        (dummy_token, dummy_hidden),
        str(output_dir / "pred_net.onnx"),
        input_names=["token", "hidden_in"],
        output_names=["output", "hidden_out"],
        opset_version=opset_version,
    )

    # Joint Network
    joint_net = JointNetwork(vocab_size=vocab_size)
    if isinstance(joint_net_state, (str, Path)):
        state = torch.load(joint_net_state, map_location="cpu", weights_only=True)
        joint_net.load_state_dict(state.get("joint_net", state))
    else:
        joint_net.load_state_dict(joint_net_state)
    joint_net.eval()

    dummy_enc = torch.randn(1, 1, 1, 384)
    dummy_pred = torch.randn(1, 1, 1, pred_net.hidden_dim)

    torch.onnx.export(
        joint_net,
        (dummy_enc, dummy_pred),
        str(output_dir / "joint_net.onnx"),
        input_names=["enc_frame", "pred_out"],
        output_names=["logits"],
        opset_version=opset_version,
    )

    pred_size = (output_dir / "pred_net.onnx").stat().st_size / 1e6
    joint_size = (output_dir / "joint_net.onnx").stat().st_size / 1e6
    print(f"Exported {language} RNN-T head to: {output_dir}")
    print(f"  pred_net.onnx: {pred_size:.2f} MB")
    print(f"  joint_net.onnx: {joint_size:.2f} MB")


def export_lid(
    model_path: str | Path,
    output_path: str | Path,
    opset_version: int = 18,
):
    """Export LID classifier to ONNX.

    Args:
        model_path: Path to LID state dict.
        output_path: Output .onnx file path.
        opset_version: ONNX opset version.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lid = MicroLID()
    lid.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    lid.eval()

    dummy = torch.randn(1, 3, 32, 128)
    batch_dim = torch.export.Dim("batch", min=1, max=64)
    width_dim = torch.export.Dim("width", min=32, max=640)

    torch.onnx.export(
        lid,
        (dummy,),
        str(output_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: batch_dim, 3: width_dim}},
        opset_version=opset_version,
    )

    print(f"Exported LID to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e3:.1f} KB")
