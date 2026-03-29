"""
LipiServer: High-throughput multilingual OCR inference server.

Uses ONNX Runtime with MultiLoRA for runtime adapter swapping.
Processes word crops in batch-by-script order for minimal adapter swaps.

Architecture:
  1. Detection model provides word bounding boxes (external)
  2. LID classifies each crop into a script family
  3. Crops are grouped by script
  4. For each script group: swap adapter, batch encode, RNN-T decode
  5. Results reassembled in page order
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


SCRIPT_MAP = {
    0: "en", 1: "hi", 2: "ta", 3: "te", 4: "kn",
    5: "bn_as", 6: "or", 7: "gu", 8: "pa", 9: "ml", 10: "ur",
}


@dataclass
class WordResult:
    """Recognition result for a single word crop."""
    text: str
    confidence: float
    script_id: str
    bbox: tuple = (0, 0, 0, 0)  # (x1, y1, x2, y2) from detection


@dataclass
class LanguagePack:
    """Loaded resources for a single language/script."""
    script_id: str
    adapter_path: str
    pred_sess: object  # ort.InferenceSession
    joint_sess: object  # ort.InferenceSession
    vocab: list[str] = field(default_factory=list)


class LipiServer:
    """High-throughput multilingual OCR inference server.

    Designed for:
      - Millions of documents
      - Batch processing (32-128 crops at a time)
      - Minimal adapter swaps (group by script, swap per-group)

    Uses ONNX Runtime MultiLoRA: one backbone in GPU memory,
    adapters swapped via LoraAdapter.Load() + RunOptions.add_active_adapter().
    """

    def __init__(self, model_dir: str, batch_size: int = 64):
        import onnxruntime as ort

        self.model_dir = Path(model_dir)
        self.batch_size = batch_size

        # Session options
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4

        providers = self._get_providers()

        # Load LID classifier
        self.lid_sess = ort.InferenceSession(
            str(self.model_dir / "lid.onnx"), sess_opts, providers=providers
        )

        # Load shared backbone
        self.backbone_sess = ort.InferenceSession(
            str(self.model_dir / "backbone.onnx"), sess_opts, providers=providers
        )

        # Load all language packs
        self.packs: dict[str, LanguagePack] = {}
        self.adapters: dict[str, object] = {}  # ort.LoraAdapter instances

        for script_id in SCRIPT_MAP.values():
            adapter_path = self.model_dir / f"adapters/{script_id}.onnx_adapter"
            pred_path = self.model_dir / f"heads/{script_id}/pred_net.onnx"
            joint_path = self.model_dir / f"heads/{script_id}/joint_net.onnx"
            vocab_path = self.model_dir / f"vocabs/{script_id}_2k.json"

            if not pred_path.exists():
                continue

            pack = LanguagePack(
                script_id=script_id,
                adapter_path=str(adapter_path),
                pred_sess=ort.InferenceSession(
                    str(pred_path), sess_opts, providers=providers
                ),
                joint_sess=ort.InferenceSession(
                    str(joint_path), sess_opts, providers=providers
                ),
                vocab=json.loads(vocab_path.read_text()) if vocab_path.exists() else [],
            )
            self.packs[script_id] = pack

            # Load adapter if available
            if adapter_path.exists():
                try:
                    self.adapters[script_id] = ort.LoraAdapter.Load(str(adapter_path))
                except AttributeError:
                    pass  # ORT version doesn't support LoraAdapter

        print(f"Loaded {len(self.packs)} language packs")

    def _get_providers(self) -> list:
        """Detect available execution providers."""
        import onnxruntime as ort
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {"device_id": 0}))
        if "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def process_page(
        self,
        word_crops: list[np.ndarray],
        bboxes: list[tuple] | None = None,
    ) -> list[WordResult]:
        """Process all word crops from a single page.

        Args:
            word_crops: List of (3, 32, W) float32 arrays.
            bboxes: Optional (x1, y1, x2, y2) tuples from detection.

        Returns:
            List of WordResult in same order as input.
        """
        n = len(word_crops)
        if n == 0:
            return []

        if bboxes is None:
            bboxes = [(0, 0, 0, 0)] * n

        # Step 1: Classify all crops by script
        script_ids = self._classify_scripts(word_crops)

        # Step 2: Group by script
        script_groups: dict[str, list[int]] = defaultdict(list)
        for i, sid in enumerate(script_ids):
            script_groups[sid].append(i)

        # Step 3: Process each script group
        results: list[WordResult | None] = [None] * n
        for script_id, indices in script_groups.items():
            if script_id not in self.packs:
                for idx in indices:
                    results[idx] = WordResult(
                        text="", confidence=0.0, script_id=script_id, bbox=bboxes[idx]
                    )
                continue

            group_crops = [word_crops[i] for i in indices]
            group_texts = self._recognize_batch(script_id, group_crops)

            for idx, (text, confidence) in zip(indices, group_texts):
                results[idx] = WordResult(
                    text=text, confidence=confidence,
                    script_id=script_id, bbox=bboxes[idx],
                )

        return results

    def _classify_scripts(self, crops: list[np.ndarray]) -> list[str]:
        """Batch LID classification."""
        max_w = max(c.shape[2] for c in crops)
        batch = np.zeros((len(crops), 3, 32, max_w), dtype=np.float32)
        for i, c in enumerate(crops):
            batch[i, :, :, :c.shape[2]] = c

        logits = self.lid_sess.run(None, {"image": batch})[0]
        return [SCRIPT_MAP.get(int(np.argmax(logits[i])), "en") for i in range(len(crops))]

    def _recognize_batch(
        self, script_id: str, crops: list[np.ndarray]
    ) -> list[tuple[str, float]]:
        """Recognize a batch of crops sharing the same script."""
        import onnxruntime as ort

        pack = self.packs[script_id]
        results = []

        for start in range(0, len(crops), self.batch_size):
            batch_crops = crops[start:start + self.batch_size]

            # Pad to same width
            max_w = max(c.shape[2] for c in batch_crops)
            batch = np.zeros((len(batch_crops), 3, 32, max_w), dtype=np.float32)
            for i, c in enumerate(batch_crops):
                batch[i, :, :, :c.shape[2]] = c

            # Encode with adapter
            run_options = ort.RunOptions()
            if script_id in self.adapters:
                try:
                    run_options.add_active_adapter(self.adapters[script_id])
                except AttributeError:
                    pass

            enc_out = self.backbone_sess.run(
                ["features", "lengths"], {"image": batch}, run_options
            )
            features, lengths = enc_out[0], enc_out[1]

            # Decode each sample
            for i in range(len(batch_crops)):
                tokens, confidence = self._greedy_decode(
                    pack, features[i:i+1], int(lengths[i])
                )
                text = "".join(
                    pack.vocab[t] for t in tokens
                    if t < len(pack.vocab)
                ) if pack.vocab else ""
                results.append((text, confidence))

        return results

    def _greedy_decode(
        self, pack: LanguagePack, enc_out: np.ndarray, T: int
    ) -> tuple[list[int], float]:
        """Greedy RNN-T decode for a single sample."""
        tokens = []
        log_probs = []
        hidden = np.zeros((1, 1, 128), dtype=np.float32)
        prev_token = np.array([[0]], dtype=np.int64)

        for t in range(T):
            if len(tokens) >= 25:
                break

            pred_out, hidden = pack.pred_sess.run(
                None, {"token": prev_token, "hidden_in": hidden}
            )

            enc_frame = enc_out[:, t:t+1, :].reshape(1, 1, 1, -1)
            pred_4d = pred_out.reshape(1, 1, 1, -1)

            logits = pack.joint_sess.run(
                None, {"enc_frame": enc_frame, "pred_out": pred_4d}
            )[0].squeeze()

            # Softmax for confidence
            exp_logits = np.exp(logits - logits.max())
            probs = exp_logits / exp_logits.sum()
            pred_id = int(np.argmax(probs))

            if pred_id == 0:
                break  # blank -> next frame
            else:
                tokens.append(pred_id)
                log_probs.append(np.log(probs[pred_id] + 1e-10))
                prev_token = np.array([[pred_id]], dtype=np.int64)

        confidence = float(np.exp(np.mean(log_probs))) if log_probs else 0.0
        return tokens, confidence
