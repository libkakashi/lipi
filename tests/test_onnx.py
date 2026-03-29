"""
ONNX export tests for all model components.

Critical Week 1 verification: if any component fails to export,
fix it before building the training pipeline.
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from src.model.encoder import LipiEncoder
from src.model.prediction_net import PredictionNetwork
from src.model.joint_net import JointNetwork
from src.model.lid import MicroLID


def _export_and_verify(model, dummy_inputs, input_names, output_names,
                       dynamic_shapes, onnx_path):
    """Export to ONNX, validate, and return ORT session."""
    import onnx
    import onnxruntime as ort

    torch.onnx.export(
        model,
        dummy_inputs,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        opset_version=18,
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    return ort.InferenceSession(onnx_path)


class TestEncoderONNXExport:

    def test_export_and_run(self):
        encoder = LipiEncoder()
        encoder.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = str(Path(tmpdir) / "encoder.onnx")
            dummy = torch.randn(1, 3, 32, 128)

            batch_dim = torch.export.Dim("batch", min=1, max=64)
            width_dim = torch.export.Dim("width", min=32, max=640)

            sess = _export_and_verify(
                encoder, (dummy,),
                input_names=["image"],
                output_names=["features", "lengths"],
                dynamic_shapes={"x": {0: batch_dim, 3: width_dim}},
                onnx_path=onnx_path,
            )

            # Verify parity
            test_input = torch.randn(1, 3, 32, 128)
            with torch.no_grad():
                pt_out, pt_len = encoder(test_input)

            ort_out, ort_len = sess.run(None, {"image": test_input.numpy()})
            np.testing.assert_allclose(pt_out.numpy(), ort_out, rtol=1e-3, atol=1e-4)


class TestPredNetONNXExport:

    def test_export_and_run(self):
        pred_net = PredictionNetwork(vocab_size=401)
        pred_net.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = str(Path(tmpdir) / "pred_net.onnx")
            dummy_token = torch.tensor([[1]], dtype=torch.long)
            dummy_hidden = torch.zeros(1, 1, 128)

            sess = _export_and_verify(
                pred_net, (dummy_token, dummy_hidden),
                input_names=["token", "hidden_in"],
                output_names=["output", "hidden_out"],
                dynamic_shapes=None,
                onnx_path=onnx_path,
            )

            # Run single step
            test_token = np.array([[5]], dtype=np.int64)
            test_hidden = np.zeros((1, 1, 128), dtype=np.float32)
            ort_out, ort_hidden = sess.run(
                None, {"token": test_token, "hidden_in": test_hidden}
            )
            assert ort_out.shape == (1, 1, 128)
            assert ort_hidden.shape == (1, 1, 128)


class TestJointNetONNXExport:

    def test_export_and_run(self):
        joint_net = JointNetwork()
        joint_net.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = str(Path(tmpdir) / "joint_net.onnx")
            dummy_enc = torch.randn(1, 1, 1, 384)
            dummy_pred = torch.randn(1, 1, 1, 128)

            sess = _export_and_verify(
                joint_net, (dummy_enc, dummy_pred),
                input_names=["enc_frame", "pred_out"],
                output_names=["logits"],
                dynamic_shapes=None,
                onnx_path=onnx_path,
            )

            # Run inference
            test_enc = np.random.randn(1, 1, 1, 384).astype(np.float32)
            test_pred = np.random.randn(1, 1, 1, 128).astype(np.float32)
            ort_logits = sess.run(
                None, {"enc_frame": test_enc, "pred_out": test_pred}
            )
            assert ort_logits[0].shape == (1, 1, 1, 401)


class TestLIDONNXExport:

    def test_export_and_run(self):
        lid = MicroLID()
        lid.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = str(Path(tmpdir) / "lid.onnx")
            dummy = torch.randn(1, 3, 32, 128)

            batch_dim = torch.export.Dim("batch", min=1, max=64)
            width_dim = torch.export.Dim("width", min=32, max=640)

            sess = _export_and_verify(
                lid, (dummy,),
                input_names=["image"],
                output_names=["logits"],
                dynamic_shapes={"x": {0: batch_dim, 3: width_dim}},
                onnx_path=onnx_path,
            )

            # Verify parity
            test_input = torch.randn(1, 3, 32, 128)
            with torch.no_grad():
                pt_logits = lid(test_input)

            ort_logits = sess.run(None, {"image": test_input.numpy()})
            np.testing.assert_allclose(
                pt_logits.numpy(), ort_logits[0], rtol=1e-3, atol=1e-4
            )


class TestAllComponentsExportTogether:
    """Verify all three components export as separate ONNX files."""

    def test_three_file_export(self):
        encoder = LipiEncoder()
        pred_net = PredictionNetwork()
        joint_net = JointNetwork()

        encoder.eval()
        pred_net.eval()
        joint_net.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Export encoder
            batch_dim = torch.export.Dim("batch", min=1, max=64)
            width_dim = torch.export.Dim("width", min=32, max=640)
            torch.onnx.export(
                encoder, (torch.randn(1, 3, 32, 128),),
                str(tmpdir / "encoder.onnx"),
                input_names=["image"],
                output_names=["features", "lengths"],
                dynamic_shapes={"x": {0: batch_dim, 3: width_dim}},
                opset_version=18,
            )

            # Export pred_net
            torch.onnx.export(
                pred_net,
                (torch.tensor([[1]], dtype=torch.long), torch.zeros(1, 1, 128)),
                str(tmpdir / "pred_net.onnx"),
                input_names=["token", "hidden_in"],
                output_names=["output", "hidden_out"],
                opset_version=18,
            )

            # Export joint_net
            torch.onnx.export(
                joint_net,
                (torch.randn(1, 1, 1, 384), torch.randn(1, 1, 1, 128)),
                str(tmpdir / "joint_net.onnx"),
                input_names=["enc_frame", "pred_out"],
                output_names=["logits"],
                opset_version=18,
            )

            # Verify all files exist
            assert (tmpdir / "encoder.onnx").exists()
            assert (tmpdir / "pred_net.onnx").exists()
            assert (tmpdir / "joint_net.onnx").exists()

            # Load all in ORT
            import onnxruntime as ort
            enc_sess = ort.InferenceSession(str(tmpdir / "encoder.onnx"))
            pred_sess = ort.InferenceSession(str(tmpdir / "pred_net.onnx"))
            joint_sess = ort.InferenceSession(str(tmpdir / "joint_net.onnx"))

            # Run a mini decode loop through all three
            test_image = np.random.randn(1, 3, 32, 128).astype(np.float32)
            enc_features, _ = enc_sess.run(None, {"image": test_image})

            # Single decode step
            hidden = np.zeros((1, 1, 128), dtype=np.float32)
            token = np.array([[0]], dtype=np.int64)
            pred_out, hidden = pred_sess.run(
                None, {"token": token, "hidden_in": hidden}
            )

            frame = enc_features[:, 0:1, :].reshape(1, 1, 1, 384)
            pred_4d = pred_out.reshape(1, 1, 1, 128)
            logits = joint_sess.run(
                None, {"enc_frame": frame, "pred_out": pred_4d}
            )

            assert logits[0].shape == (1, 1, 1, 401)
