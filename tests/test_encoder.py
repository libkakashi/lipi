"""
Checkpoint 0 tests for the encoder backbone.

Tests:
  - Correct output shapes for various widths
  - Batch size handling
  - Gradient flow (backward pass)
  - ONNX export and inference parity
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from src.model.encoder import LipiEncoder


@pytest.fixture
def encoder():
    model = LipiEncoder()
    model.eval()
    return model


def _export_encoder_onnx(encoder, onnx_path: str):
    """Helper to export encoder to ONNX with correct settings."""
    dummy = torch.randn(1, 3, 32, 128)

    # Use dynamic_shapes (torch 2.11+ preferred over dynamic_axes)
    batch = torch.export.Dim("batch", min=1, max=64)
    width = torch.export.Dim("width", min=32, max=640)
    dynamic_shapes = {
        "x": {0: batch, 3: width},
    }

    torch.onnx.export(
        encoder,
        (dummy,),
        onnx_path,
        input_names=["image"],
        output_names=["features", "lengths"],
        dynamic_shapes=dynamic_shapes,
        opset_version=18,
    )


class TestEncoderShapes:
    """Verify encoder produces correct output shapes."""

    @pytest.mark.parametrize("width", [32, 64, 128, 256, 320])
    def test_output_shape_various_widths(self, encoder, width):
        x = torch.randn(1, 3, 32, width)
        features, lengths = encoder(x)
        expected_T = width // 4
        assert features.shape == (1, expected_T, 384)
        assert lengths.shape == (1,)
        assert lengths.item() == expected_T

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_output_shape_various_batches(self, encoder, batch_size):
        x = torch.randn(batch_size, 3, 32, 128)
        features, lengths = encoder(x)
        assert features.shape == (batch_size, 32, 384)
        assert lengths.shape == (batch_size,)

    def test_output_dim_attribute(self, encoder):
        assert encoder.output_dim == 384


class TestEncoderGradients:
    """Verify gradients flow through the entire encoder."""

    def test_backward_pass(self):
        encoder = LipiEncoder()
        encoder.train()
        x = torch.randn(2, 3, 32, 128)
        features, lengths = encoder(x)
        loss = features.sum()
        loss.backward()

        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestEncoderONNX:
    """Verify ONNX export and inference parity."""

    def test_onnx_export(self, encoder):
        """Export encoder to ONNX and verify it loads."""
        import onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = str(Path(tmpdir) / "encoder.onnx")
            _export_encoder_onnx(encoder, onnx_path)

            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)

    def test_onnx_inference_parity(self, encoder):
        """Verify ONNX runtime output matches PyTorch output."""
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = str(Path(tmpdir) / "encoder.onnx")
            _export_encoder_onnx(encoder, onnx_path)

            # PyTorch inference
            test_input = torch.randn(1, 3, 32, 128)
            with torch.no_grad():
                pt_features, pt_lengths = encoder(test_input)

            # ONNX Runtime inference
            sess = ort.InferenceSession(onnx_path)
            ort_features, ort_lengths = sess.run(
                None, {"image": test_input.numpy()}
            )

            np.testing.assert_allclose(
                pt_features.numpy(), ort_features, rtol=1e-3, atol=1e-4
            )
            np.testing.assert_array_equal(pt_lengths.numpy(), ort_lengths)

    def test_onnx_dynamic_width(self, encoder):
        """Verify ONNX model handles dynamic widths correctly."""
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = str(Path(tmpdir) / "encoder.onnx")
            _export_encoder_onnx(encoder, onnx_path)

            sess = ort.InferenceSession(onnx_path)

            for width in [32, 64, 128, 256]:
                test_input = np.random.randn(1, 3, 32, width).astype(np.float32)
                ort_features, ort_lengths = sess.run(None, {"image": test_input})
                expected_T = width // 4
                assert ort_features.shape == (1, expected_T, 384), (
                    f"W={width}: got {ort_features.shape}"
                )
