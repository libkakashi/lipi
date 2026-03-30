#!/usr/bin/env python3
"""
ONNX Export Verification Script.

Run this in Week 1 IMMEDIATELY after implementing the encoder.
If any component fails to export, fix attention.py NOW.

Tests:
  1. Export encoder to ONNX
  2. Export pred_net to ONNX
  3. Export joint_net to ONNX
  4. Export LID to ONNX
  5. Load all in ORT and run inference
  6. Verify PyTorch-ORT parity
  7. Test dynamic width handling
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.encoder import LipiEncoder
from src.model.prediction_net import PredictionNetwork
from src.model.joint_net import JointNetwork
from src.model.lid import MicroLID


def test_encoder_export():
    print("1. Testing encoder ONNX export...")
    import onnx
    import onnxruntime as ort

    encoder = LipiEncoder()
    encoder.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "encoder.onnx")
        dummy = torch.randn(1, 3, 32, 128)

        batch_dim = torch.export.Dim("batch", min=1, max=64)
        width_dim = torch.export.Dim("width", min=32, max=640)

        torch.onnx.export(
            encoder, (dummy,), path,
            input_names=["image"],
            output_names=["features", "lengths"],
            dynamic_shapes={"x": {0: batch_dim, 3: width_dim}},
            opset_version=18,
        )

        # Validate
        model = onnx.load(path)
        onnx.checker.check_model(model)

        # Parity check
        test = torch.randn(1, 3, 32, 128)
        with torch.no_grad():
            pt_out, pt_len = encoder(test)

        sess = ort.InferenceSession(path)
        ort_out, ort_len = sess.run(None, {"image": test.numpy()})
        np.testing.assert_allclose(pt_out.numpy(), ort_out, rtol=1e-3, atol=1e-4)

        # Dynamic width
        for w in [32, 64, 128, 256]:
            test_w = np.random.randn(1, 3, 32, w).astype(np.float32)
            out, _ = sess.run(None, {"image": test_w})
            assert out.shape == (1, w // 4, 384), f"W={w}: got {out.shape}"

        size_mb = Path(path).stat().st_size / 1e6
        print(f"   PASS — encoder.onnx ({size_mb:.1f} MB), dynamic width OK")


def test_pred_net_export():
    print("2. Testing pred_net ONNX export...")
    import onnxruntime as ort

    pred_net = PredictionNetwork(vocab_size=171)
    pred_net.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "pred_net.onnx")
        torch.onnx.export(
            pred_net,
            (torch.tensor([[1]], dtype=torch.long), torch.zeros(1, 1, 128)),
            path,
            input_names=["token", "hidden_in"],
            output_names=["output", "hidden_out"],
            opset_version=18,
        )

        sess = ort.InferenceSession(path)
        out, hidden = sess.run(None, {
            "token": np.array([[5]], dtype=np.int64),
            "hidden_in": np.zeros((1, 1, 128), dtype=np.float32),
        })
        assert out.shape == (1, 1, 128)
        assert hidden.shape == (1, 1, 128)

        size_kb = Path(path).stat().st_size / 1e3
        print(f"   PASS — pred_net.onnx ({size_kb:.0f} KB)")


def test_joint_net_export():
    print("3. Testing joint_net ONNX export...")
    import onnxruntime as ort

    joint_net = JointNetwork(vocab_size=171)
    joint_net.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "joint_net.onnx")
        torch.onnx.export(
            joint_net,
            (torch.randn(1, 1, 1, 384), torch.randn(1, 1, 1, 128)),
            path,
            input_names=["enc_frame", "pred_out"],
            output_names=["logits"],
            opset_version=18,
        )

        sess = ort.InferenceSession(path)
        logits = sess.run(None, {
            "enc_frame": np.random.randn(1, 1, 1, 384).astype(np.float32),
            "pred_out": np.random.randn(1, 1, 1, 128).astype(np.float32),
        })
        assert logits[0].shape == (1, 1, 1, 171)

        size_kb = Path(path).stat().st_size / 1e3
        print(f"   PASS — joint_net.onnx ({size_kb:.0f} KB)")


def test_lid_export():
    print("4. Testing LID ONNX export...")
    import onnxruntime as ort

    lid = MicroLID()
    lid.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "lid.onnx")

        batch_dim = torch.export.Dim("batch", min=1, max=64)
        width_dim = torch.export.Dim("width", min=32, max=640)

        torch.onnx.export(
            lid, (torch.randn(1, 3, 32, 128),), path,
            input_names=["image"],
            output_names=["logits"],
            dynamic_shapes={"x": {0: batch_dim, 3: width_dim}},
            opset_version=18,
        )

        sess = ort.InferenceSession(path)
        logits = sess.run(None, {"image": np.random.randn(1, 3, 32, 128).astype(np.float32)})
        assert logits[0].shape == (1, 11)

        size_kb = Path(path).stat().st_size / 1e3
        print(f"   PASS — lid.onnx ({size_kb:.0f} KB)")


def test_end_to_end_ort():
    print("5. Testing end-to-end ORT decode loop...")
    import onnxruntime as ort

    encoder = LipiEncoder()
    pred_net = PredictionNetwork(vocab_size=171)
    joint_net = JointNetwork(vocab_size=171)

    encoder.eval()
    pred_net.eval()
    joint_net.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        batch_dim = torch.export.Dim("batch", min=1, max=64)
        width_dim = torch.export.Dim("width", min=32, max=640)

        torch.onnx.export(
            encoder, (torch.randn(1, 3, 32, 128),),
            str(tmpdir / "enc.onnx"),
            input_names=["image"], output_names=["features", "lengths"],
            dynamic_shapes={"x": {0: batch_dim, 3: width_dim}},
            opset_version=18,
        )
        torch.onnx.export(
            pred_net,
            (torch.tensor([[1]], dtype=torch.long), torch.zeros(1, 1, 128)),
            str(tmpdir / "pred.onnx"),
            input_names=["token", "hidden_in"],
            output_names=["output", "hidden_out"],
            opset_version=18,
        )
        torch.onnx.export(
            joint_net,
            (torch.randn(1, 1, 1, 384), torch.randn(1, 1, 1, 128)),
            str(tmpdir / "joint.onnx"),
            input_names=["enc_frame", "pred_out"],
            output_names=["logits"],
            opset_version=18,
        )

        enc_sess = ort.InferenceSession(str(tmpdir / "enc.onnx"))
        pred_sess = ort.InferenceSession(str(tmpdir / "pred.onnx"))
        joint_sess = ort.InferenceSession(str(tmpdir / "joint.onnx"))

        # Run one full decode step
        image = np.random.randn(1, 3, 32, 128).astype(np.float32)
        features, lengths = enc_sess.run(None, {"image": image})

        hidden = np.zeros((1, 1, 128), dtype=np.float32)
        token = np.array([[0]], dtype=np.int64)

        pred_out, hidden = pred_sess.run(None, {"token": token, "hidden_in": hidden})
        frame = features[:, 0:1, :].reshape(1, 1, 1, 384)
        pred_4d = pred_out.reshape(1, 1, 1, 128)

        logits = joint_sess.run(None, {"enc_frame": frame, "pred_out": pred_4d})
        pred_id = int(np.argmax(logits[0].squeeze()))

        print(f"   PASS — Full decode step: predicted token_id={pred_id}")


if __name__ == "__main__":
    print("=" * 60)
    print("ONNX EXPORT VERIFICATION")
    print("=" * 60)
    print()

    test_encoder_export()
    test_pred_net_export()
    test_joint_net_export()
    test_lid_export()
    test_end_to_end_ort()

    print()
    print("=" * 60)
    print("ALL ONNX EXPORT TESTS PASSED")
    print("=" * 60)
