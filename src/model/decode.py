"""
RNN-T Greedy Decoding.

Decodes encoder output into token sequences using greedy search.
For each encoder frame, repeatedly queries the joint network until
it emits a blank (advance to next frame) or reaches max emissions.
"""

import torch
from torch import Tensor

from src.model.prediction_net import PredictionNetwork
from src.model.joint_net import JointNetwork


@torch.no_grad()
def greedy_decode(
    enc_out: Tensor,
    pred_net: PredictionNetwork,
    joint_net: JointNetwork,
    max_tokens: int = 25,
    blank_id: int = 0,
) -> list[list[int]]:
    """Greedy RNN-T decode for a batch of encoder outputs.

    Args:
        enc_out: (B, T, enc_dim) — encoder features.
        pred_net: Prediction network.
        joint_net: Joint network.
        max_tokens: Maximum number of non-blank tokens per sample.
        blank_id: Token ID for blank symbol.

    Returns:
        List of B token ID lists (excluding blank).
    """
    B, T, _ = enc_out.shape
    device = enc_out.device
    results: list[list[int]] = []

    for b in range(B):
        tokens: list[int] = []
        hidden: Tensor | None = None
        prev_token = torch.tensor([[blank_id]], dtype=torch.long, device=device)

        for t in range(T):
            # Encoder frame: (1, 1, 1, enc_dim)
            frame = enc_out[b, t].reshape(1, 1, 1, -1)

            # Allow multiple emissions per frame (up to remaining budget)
            max_per_frame = max_tokens - len(tokens)
            for _ in range(max(max_per_frame, 1)):
                # Prediction step
                pred_out, hidden = pred_net(prev_token, hidden)
                # pred_out: (1, 1, hidden_dim) -> (1, 1, 1, hidden_dim)
                pred_out_4d = pred_out.unsqueeze(1)

                # Joint step
                logits = joint_net(frame, pred_out_4d)  # (1, 1, 1, vocab_size)
                pred_id = logits.squeeze().argmax().item()

                if pred_id == blank_id:
                    break  # Advance to next encoder frame
                else:
                    tokens.append(pred_id)
                    prev_token = torch.tensor(
                        [[pred_id]], dtype=torch.long, device=device
                    )
                    if len(tokens) >= max_tokens:
                        break

            if len(tokens) >= max_tokens:
                break

        results.append(tokens)

    return results


@torch.no_grad()
def greedy_decode_with_confidence(
    enc_out: Tensor,
    pred_net: PredictionNetwork,
    joint_net: JointNetwork,
    max_tokens: int = 25,
    blank_id: int = 0,
) -> list[tuple[list[int], float]]:
    """Greedy decode with confidence scores.

    Confidence = geometric mean of token probabilities from joint network.

    Args:
        enc_out: (B, T, enc_dim)
        pred_net: Prediction network.
        joint_net: Joint network.
        max_tokens: Maximum tokens per sample.
        blank_id: Blank token ID.

    Returns:
        List of (token_ids, confidence) tuples.
    """
    import torch.nn.functional as F

    B, T, _ = enc_out.shape
    device = enc_out.device
    results: list[tuple[list[int], float]] = []

    for b in range(B):
        tokens: list[int] = []
        log_probs: list[float] = []
        hidden: Tensor | None = None
        prev_token = torch.tensor([[blank_id]], dtype=torch.long, device=device)

        for t in range(T):
            frame = enc_out[b, t].reshape(1, 1, 1, -1)

            max_per_frame = max_tokens - len(tokens)
            for _ in range(max(max_per_frame, 1)):
                pred_out, hidden = pred_net(prev_token, hidden)
                pred_out_4d = pred_out.unsqueeze(1)

                logits = joint_net(frame, pred_out_4d).squeeze()
                probs = F.softmax(logits, dim=-1)
                pred_id = probs.argmax().item()

                if pred_id == blank_id:
                    break
                else:
                    tokens.append(pred_id)
                    log_probs.append(torch.log(probs[pred_id] + 1e-10).item())
                    prev_token = torch.tensor(
                        [[pred_id]], dtype=torch.long, device=device
                    )
                    if len(tokens) >= max_tokens:
                        break

            if len(tokens) >= max_tokens:
                break

        if log_probs:
            confidence = float(torch.exp(torch.tensor(log_probs).mean()))
        else:
            confidence = 0.0

        results.append((tokens, confidence))

    return results
