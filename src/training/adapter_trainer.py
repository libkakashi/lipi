"""
Phase 2: LoRA Adapter Training.

Freezes the backbone, injects LoRA into Stage 2 + Stage 3,
attaches a fresh RNN-T head, and trains LoRA + head on
language-specific data.

Only trainable params: LoRA weights + prediction_net + joint_net.
Backbone optimizer state is NOT created (saves GPU memory).
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
from src.model.prediction_net import PredictionNetwork
from src.model.joint_net import JointNetwork
from src.model.lora import inject_lora
from src.model.decode import greedy_decode
from src.data.dataset import collate_ocr
from src.data.bigrams import LipiTokenizer
from src.training.loss import rnnt_loss


class AdapterTrainer:
    """Phase 2: LoRA adapter + RNN-T head training."""

    def __init__(
        self,
        backbone: LipiEncoder,
        tokenizer: LipiTokenizer,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        pred_embed_dim: int = 128,
        pred_hidden_dim: int = 128,
        joint_dim: int = 256,
        lora_lr: float = 1e-3,
        head_lr: float = 1e-3,
        weight_decay: float = 0.01,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.tokenizer = tokenizer
        vocab_size = tokenizer.vocab_size

        # Inject LoRA into backbone (freezes everything, enables LoRA params)
        self.model = inject_lora(
            backbone, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout
        ).to(self.device)

        # Fresh RNN-T head
        self.pred_net = PredictionNetwork(
            vocab_size=vocab_size,
            embed_dim=pred_embed_dim,
            hidden_dim=pred_hidden_dim,
        ).to(self.device)

        self.joint_net = JointNetwork(
            enc_dim=backbone.output_dim,
            pred_dim=pred_hidden_dim,
            joint_dim=joint_dim,
            vocab_size=vocab_size,
        ).to(self.device)

        # Optimizer: only LoRA params + head params
        # Do NOT include frozen backbone params (saves optimizer state memory)
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        head_params = list(self.pred_net.parameters()) + list(self.joint_net.parameters())

        self.optimizer = AdamW([
            {"params": lora_params, "lr": lora_lr},
            {"params": head_params, "lr": head_lr},
        ], weight_decay=weight_decay)

        self.scheduler = None  # Set in train()

    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 15,
        save_dir: str | Path = "checkpoints",
        log_interval: int = 50,
        eval_fn=None,
    ) -> dict:
        """Run Phase 2 adapter training.

        Args:
            train_loader: DataLoader yielding (images, labels, widths).
            epochs: Number of training epochs.
            save_dir: Directory to save checkpoints.
            log_interval: Steps between log messages.
            eval_fn: Optional evaluation function called after each epoch.

        Returns:
            Dict with training history.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        total_steps = len(train_loader) * epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[g["lr"] for g in self.optimizer.param_groups],
            total_steps=total_steps,
            pct_start=0.1,
        )

        self.model.train()
        self.pred_net.train()
        self.joint_net.train()

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

                max_target_len = max(target_lengths) if target_lengths else 1
                targets = torch.zeros(len(labels), max_target_len, dtype=torch.long)
                for i, ids in enumerate(target_ids):
                    targets[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                targets = targets.to(self.device, non_blocking=True)
                target_lengths_t = torch.tensor(
                    target_lengths, dtype=torch.long, device=self.device
                )

                # Forward with mixed precision
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    features, _ = self.model(images)
                    B = targets.shape[0]
                    T = features.shape[1]
                    enc_lengths = torch.full(
                        (B,), T, dtype=torch.long, device=self.device
                    )
                    blank = torch.zeros(B, 1, dtype=torch.long, device=self.device)
                    pred_input = torch.cat([blank, targets], dim=1)

                    pred_out, _ = self.pred_net(pred_input)

                    enc_expanded = features.unsqueeze(2)
                    pred_expanded = pred_out.unsqueeze(1)
                    logits = self.joint_net(enc_expanded, pred_expanded)

                # RNN-T loss (outside autocast — needs float32)
                loss = rnnt_loss(
                    logits.float(), targets, enc_lengths, target_lengths_t,
                    blank=self.tokenizer.blank_id,
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) +
                    list(self.pred_net.parameters()) +
                    list(self.joint_net.parameters()),
                    max_norm=1.0,
                )
                scaler.step(self.optimizer)
                scaler.update()
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

            if eval_fn:
                eval_fn(epoch + 1)

            self.save_checkpoint(save_dir / f"adapter_epoch{epoch+1}.pt")

        return history

    @torch.no_grad()
    def evaluate(
        self,
        eval_loader: DataLoader,
        max_batches: int | None = None,
    ) -> dict:
        """Evaluate word accuracy using greedy decode.

        Args:
            eval_loader: DataLoader yielding (images, labels, widths).
            max_batches: Limit evaluation to N batches (for speed).

        Returns:
            Dict with accuracy metrics.
        """
        self.model.eval()
        self.pred_net.eval()
        self.joint_net.eval()

        correct = 0
        total = 0

        for i, (images, labels, widths) in enumerate(eval_loader):
            if max_batches and i >= max_batches:
                break

            images = images.to(self.device)
            features, enc_lengths = self.model(images)

            decoded = greedy_decode(
                features, self.pred_net, self.joint_net,
                max_tokens=25, blank_id=self.tokenizer.blank_id,
            )

            for pred_ids, true_label in zip(decoded, labels):
                pred_text = self.tokenizer.decode(pred_ids)
                if pred_text == true_label:
                    correct += 1
                total += 1

        self.model.train()
        self.pred_net.train()
        self.joint_net.train()

        accuracy = correct / max(total, 1)
        return {"accuracy": accuracy, "correct": correct, "total": total}

    def save_checkpoint(self, path: str | Path):
        """Save LoRA weights + RNN-T head."""
        # Extract LoRA state dict (only LoRA params, not frozen backbone)
        lora_state = {
            k: v for k, v in self.model.state_dict().items()
            if "lora_" in k
        }

        torch.save({
            "lora": lora_state,
            "pred_net": self.pred_net.state_dict(),
            "joint_net": self.joint_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
