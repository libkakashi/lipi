"""
Phase 1: Foundation Backbone Training.

Trains the full backbone with a temporary CTC head on Pan-Indic data.
The CTC head is discarded after training — only the backbone is kept.

Uses CTC loss (simpler, faster convergence) rather than RNN-T because
we just need the backbone to learn visual features. The sophisticated
RNN-T decoding comes in Phase 2 with language-specific heads.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
from tqdm import tqdm

from src.model.encoder import LipiEncoder
from src.data.dataset import collate_ocr
from src.data.bigrams import LipiTokenizer
from src.training.loss import ctc_loss


class CTCHead(nn.Module):
    """Simple CTC head — single linear projection. 37K params."""

    def __init__(self, enc_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(enc_dim, vocab_size)

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, T, enc_dim) from encoder.

        Returns:
            (B, T, vocab_size) — logits for CTC.
        """
        return self.proj(features)


class CTCHeadMLP(nn.Module):
    """MLP CTC head — more capacity than single linear. ~135K params."""

    def __init__(self, enc_dim: int, vocab_size: int, hidden_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.head(features)


class CTCHeadBiLSTM(nn.Module):
    """BiLSTM CTC head — bidirectional context for each frame. ~430K params.

    Gives CTC the sequential context that RNN-T gets from the GRU,
    but runs in parallel (no autoregressive decode loop).
    Still 4-5x faster than RNN-T at inference.
    """

    def __init__(self, enc_dim: int, vocab_size: int, hidden_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(enc_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, features: Tensor) -> Tensor:
        out, _ = self.lstm(features)
        return self.proj(out)


class FoundationTrainer:
    """Phase 1 training loop.

    Trains encoder + CTC head on multi-script data with character-level CTC loss.
    """

    def __init__(
        self,
        encoder: LipiEncoder,
        tokenizer: LipiTokenizer,
        lr: float = 7e-4,
        weight_decay: float = 0.01,
        warmup_pct: float = 0.1,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device)
        self.ctc_head = CTCHead(encoder.output_dim, tokenizer.vocab_size).to(self.device)
        self.tokenizer = tokenizer

        # Only train encoder + CTC head params
        self.params = list(self.encoder.parameters()) + list(self.ctc_head.parameters())
        self.optimizer = AdamW(self.params, lr=lr, weight_decay=weight_decay)
        self.scheduler = None  # Set in train()
        self.warmup_pct = warmup_pct

    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 20,
        save_dir: str | Path = "checkpoints",
        log_interval: int = 50,
    ) -> dict:
        """Run Phase 1 training.

        Args:
            train_loader: DataLoader yielding (images, labels, widths).
            epochs: Number of training epochs.
            save_dir: Directory to save checkpoints.
            log_interval: Steps between log messages.

        Returns:
            Dict with training history.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        total_steps = len(train_loader) * epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.defaults["lr"],
            total_steps=total_steps,
            pct_start=self.warmup_pct,
        )

        self.encoder.train()
        self.ctc_head.train()

        # Mixed precision for CUDA
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

        history = {"loss": [], "epoch_loss": []}
        global_step = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, labels, widths in pbar:
                images = images.to(self.device, non_blocking=True)

                # Encode targets
                target_ids = []
                target_lengths = []
                for label in labels:
                    ids = self.tokenizer.encode(label)
                    target_ids.append(ids)
                    target_lengths.append(len(ids))

                # Pad targets
                max_target_len = max(target_lengths) if target_lengths else 1
                targets = torch.zeros(len(labels), max_target_len, dtype=torch.long)
                for i, ids in enumerate(target_ids):
                    targets[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                targets = targets.to(self.device, non_blocking=True)
                target_lengths_t = torch.tensor(target_lengths, dtype=torch.long, device=self.device)

                # Forward pass with mixed precision
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    features, enc_lengths = self.encoder(images)
                    logits = self.ctc_head(features)

                # CTC loss (outside autocast — needs float32)
                loss = ctc_loss(
                    logits.float(), targets,
                    enc_lengths, target_lengths_t,
                    blank=self.tokenizer.blank_id,
                )

                if torch.isinf(loss):
                    continue

                # Backward + optimize with scaler
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
                old_scale = scaler.get_scale()
                scaler.step(self.optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    self.scheduler.step()

                loss_val = loss.item()
                history["loss"].append(loss_val)
                epoch_loss += loss_val
                num_batches += 1
                global_step += 1

                if global_step % log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}")

            avg_loss = epoch_loss / max(num_batches, 1)
            history["epoch_loss"].append(avg_loss)
            print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

            # Save checkpoint
            self.save_checkpoint(save_dir / f"phase1_epoch{epoch+1}.pt")

        return history

    def save_checkpoint(self, path: str | Path):
        """Save encoder weights (backbone only, not CTC head)."""
        torch.save({
            "encoder": self.encoder.state_dict(),
            "ctc_head": self.ctc_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    @staticmethod
    def load_backbone(path: str | Path, device: str = "cpu") -> LipiEncoder:
        """Load just the encoder backbone from a Phase 1 checkpoint."""
        encoder = LipiEncoder()
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        encoder.load_state_dict(checkpoint["encoder"])
        return encoder
