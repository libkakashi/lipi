"""
Checkpoint 1: Overfit Test v2.

Uses the synthetic renderer with diverse fonts to create visually
distinct training samples. Tests both CTC (Phase 1) and RNN-T (Phase 2) paths.

Run on M3 Pro with MPS fallback for RNN-T loss.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.encoder import LipiEncoder
from src.model.prediction_net import PredictionNetwork
from src.model.joint_net import JointNetwork
from src.model.decode import greedy_decode
from src.training.foundation_trainer import CTCHead
from src.training.loss import rnnt_loss, ctc_loss
from src.data.bigrams import LipiTokenizer
from src.data.synth import generate_dataset
from src.data.dataset import InMemoryDataset, collate_ocr


def ctc_greedy_decode(logits, tokenizer):
    """CTC greedy decode: argmax, collapse repeats, remove blanks."""
    preds = logits.argmax(dim=-1)  # (B, T)
    results = []
    for i in range(preds.shape[0]):
        p = preds[i].tolist()
        collapsed = [p[0]] + [p[j] for j in range(1, len(p)) if p[j] != p[j-1]]
        collapsed = [x for x in collapsed if x != 0]
        results.append(tokenizer.decode(collapsed))
    return results


def run_ctc_overfit(
    words, n_per_word=20, num_steps=3000, lr=3e-3, batch_size=16, device_str="cpu"
):
    """Test CTC path (Phase 1 style) with diverse synthetic data."""
    device = torch.device(device_str)
    print(f"\n{'='*60}")
    print(f"CTC OVERFIT TEST: {len(words)} words × {n_per_word} variants = {len(words)*n_per_word} samples")
    print(f"Device: {device}, LR: {lr}, Steps: {num_steps}, Batch: {batch_size}")
    print(f"{'='*60}\n")

    # Generate diverse synthetic data
    print("Generating synthetic images with diverse fonts...")
    images, labels = generate_dataset(words, n_per_word=n_per_word)
    print(f"Generated {len(images)} images")

    tokenizer = LipiTokenizer.build_character_level("en")
    encoder = LipiEncoder().to(device)
    ctc_head = CTCHead(encoder.output_dim, tokenizer.vocab_size).to(device)

    params = list(encoder.parameters()) + list(ctc_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    dataset = InMemoryDataset(images, labels)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_ocr, drop_last=True,
    )

    encoder.train()
    ctc_head.train()

    step = 0
    start_time = time.time()
    loader_iter = iter(loader)

    while step < num_steps:
        try:
            batch_imgs, batch_labels, widths = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch_imgs, batch_labels, widths = next(loader_iter)

        batch_imgs = batch_imgs.to(device)

        target_ids = [tokenizer.encode(l) for l in batch_labels]
        target_lengths = torch.tensor([len(ids) for ids in target_ids])
        max_tgt = max(len(ids) for ids in target_ids) if target_ids else 1
        targets = torch.zeros(len(batch_labels), max_tgt, dtype=torch.long)
        for i, ids in enumerate(target_ids):
            targets[i, :len(ids)] = torch.tensor(ids)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        features, enc_lengths = encoder(batch_imgs)
        logits = ctc_head(features)

        loss = ctc_loss(logits, targets, enc_lengths, target_lengths)

        if torch.isinf(loss) or torch.isnan(loss):
            step += 1
            continue

        optimizer.zero_grad()
        loss.backward()
        if device.type == "mps":
            torch.mps.synchronize()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()

        step += 1

        if step % 100 == 0 or step == 1:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            # Quick decode check on current batch
            encoder.eval()
            ctc_head.eval()
            with torch.no_grad():
                test_logits = ctc_head(encoder(batch_imgs)[0])
                decoded = ctc_greedy_decode(test_logits, tokenizer)
                batch_correct = sum(1 for d, l in zip(decoded, batch_labels) if d == l)
            encoder.train()
            ctc_head.train()
            print(
                f"Step {step:5d}: loss={loss.item():.4f}, "
                f"batch_acc={batch_correct}/{len(batch_labels)}, "
                f"speed={steps_per_sec:.1f} step/s, "
                f"sample: \"{batch_labels[0]}\" -> \"{decoded[0]}\""
            )

    # Full evaluation
    print(f"\n{'='*60}")
    print("FULL EVALUATION")
    print(f"{'='*60}")

    encoder.eval()
    ctc_head.eval()

    eval_loader = DataLoader(
        dataset, batch_size=32, shuffle=False, collate_fn=collate_ocr,
    )

    per_word_correct = {w: 0 for w in words}
    per_word_total = {w: 0 for w in words}

    with torch.no_grad():
        for batch_imgs, batch_labels, widths in eval_loader:
            batch_imgs = batch_imgs.to(device)
            features, enc_lengths = encoder(batch_imgs)
            logits = ctc_head(features)
            decoded = ctc_greedy_decode(logits, tokenizer)

            for d, l in zip(decoded, batch_labels):
                per_word_total[l] += 1
                if d == l:
                    per_word_correct[l] += 1

    total_correct = sum(per_word_correct.values())
    total = sum(per_word_total.values())

    for w in words:
        acc = per_word_correct[w] / max(per_word_total[w], 1) * 100
        print(f"  {w:10s}: {per_word_correct[w]:3d}/{per_word_total[w]:3d} ({acc:.0f}%)")

    overall_acc = total_correct / max(total, 1) * 100
    print(f"\n  OVERALL: {total_correct}/{total} ({overall_acc:.1f}%)")
    print(f"  Final loss: {loss.item():.4f}")
    print(f"  Total time: {time.time() - start_time:.0f}s")

    return overall_acc, loss.item()


def run_rnnt_overfit(
    words, n_per_word=20, num_steps=3000, lr=1e-3, batch_size=16, device_str="cpu"
):
    """Test RNN-T path (Phase 2 style) with diverse synthetic data."""
    device = torch.device(device_str)
    print(f"\n{'='*60}")
    print(f"RNN-T OVERFIT TEST: {len(words)} words × {n_per_word} variants = {len(words)*n_per_word} samples")
    print(f"Device: {device}, LR: {lr}, Steps: {num_steps}, Batch: {batch_size}")
    print(f"{'='*60}\n")

    print("Generating synthetic images...")
    images, labels = generate_dataset(words, n_per_word=n_per_word)

    tokenizer = LipiTokenizer.build_character_level("en")
    encoder = LipiEncoder().to(device)
    pred_net = PredictionNetwork(vocab_size=tokenizer.vocab_size).to(device)
    joint_net = JointNetwork(
        enc_dim=encoder.output_dim,
        pred_dim=pred_net.hidden_dim,
        joint_dim=256,
        vocab_size=tokenizer.vocab_size,
    ).to(device)

    all_params = (
        list(encoder.parameters()) +
        list(pred_net.parameters()) +
        list(joint_net.parameters())
    )
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.0)

    dataset = InMemoryDataset(images, labels)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_ocr, drop_last=True,
    )

    encoder.train()
    pred_net.train()
    joint_net.train()

    step = 0
    start_time = time.time()
    loader_iter = iter(loader)

    while step < num_steps:
        try:
            batch_imgs, batch_labels, widths = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch_imgs, batch_labels, widths = next(loader_iter)

        batch_imgs = batch_imgs.to(device)

        target_ids = [tokenizer.encode(l) for l in batch_labels]
        target_lengths = torch.tensor([len(ids) for ids in target_ids], device=device)
        max_tgt = max(len(ids) for ids in target_ids) if target_ids else 1
        targets = torch.zeros(len(batch_labels), max_tgt, dtype=torch.long, device=device)
        for i, ids in enumerate(target_ids):
            targets[i, :len(ids)] = torch.tensor(ids)

        features, _ = encoder(batch_imgs)
        B = targets.shape[0]
        T = features.shape[1]
        enc_lengths = torch.full((B,), T, dtype=torch.long, device=device)

        blank = torch.zeros(B, 1, dtype=torch.long, device=device)
        pred_input = torch.cat([blank, targets], dim=1)
        pred_out, _ = pred_net(pred_input)

        enc_expanded = features.unsqueeze(2)
        pred_expanded = pred_out.unsqueeze(1)
        logits = joint_net(enc_expanded, pred_expanded)

        loss = rnnt_loss(logits, targets, enc_lengths, target_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            step += 1
            continue

        optimizer.zero_grad()
        loss.backward()
        if device.type == "mps":
            torch.mps.synchronize()
        torch.nn.utils.clip_grad_norm_(all_params, 5.0)
        optimizer.step()

        step += 1

        if step % 100 == 0 or step == 1:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed

            # Quick decode check
            encoder.eval()
            pred_net.eval()
            joint_net.eval()
            with torch.no_grad():
                test_features, _ = encoder(batch_imgs[:4])
                decoded = greedy_decode(test_features, pred_net, joint_net, max_tokens=15)
                decoded_text = [tokenizer.decode(d) for d in decoded]
                batch_correct = sum(
                    1 for d, l in zip(decoded_text, batch_labels[:4]) if d == l
                )
            encoder.train()
            pred_net.train()
            joint_net.train()

            print(
                f"Step {step:5d}: loss={loss.item():.4f}, "
                f"sample_acc={batch_correct}/4, "
                f"speed={steps_per_sec:.1f} step/s, "
                f"sample: \"{batch_labels[0]}\" -> \"{decoded_text[0]}\""
            )

    # Full evaluation
    print(f"\n{'='*60}")
    print("FULL EVALUATION")
    print(f"{'='*60}")

    encoder.eval()
    pred_net.eval()
    joint_net.eval()

    eval_loader = DataLoader(
        dataset, batch_size=32, shuffle=False, collate_fn=collate_ocr,
    )

    per_word_correct = {w: 0 for w in words}
    per_word_total = {w: 0 for w in words}

    with torch.no_grad():
        for batch_imgs, batch_labels, widths in eval_loader:
            batch_imgs = batch_imgs.to(device)
            features, _ = encoder(batch_imgs)
            decoded = greedy_decode(features, pred_net, joint_net, max_tokens=15)

            for d, l in zip(decoded, batch_labels):
                text = tokenizer.decode(d)
                per_word_total[l] += 1
                if text == l:
                    per_word_correct[l] += 1

    total_correct = sum(per_word_correct.values())
    total = sum(per_word_total.values())

    for w in words:
        acc = per_word_correct[w] / max(per_word_total[w], 1) * 100
        print(f"  {w:10s}: {per_word_correct[w]:3d}/{per_word_total[w]:3d} ({acc:.0f}%)")

    overall_acc = total_correct / max(total, 1) * 100
    print(f"\n  OVERALL: {total_correct}/{total} ({overall_acc:.1f}%)")
    print(f"  Final loss: {loss.item():.4f}")
    print(f"  Total time: {time.time() - start_time:.0f}s")

    return overall_acc, loss.item()


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Test words — short, visually distinct
    words = ["Hello", "World", "Code", "Data", "Test"]

    # Phase 1 style: CTC
    ctc_acc, ctc_loss_val = run_ctc_overfit(
        words, n_per_word=20, num_steps=3000, lr=3e-3,
        batch_size=16, device_str=device,
    )

    # Phase 2 style: RNN-T (only if CTC shows the encoder learns)
    if ctc_acc > 30:
        print("\n\nCTC shows learning — now testing RNN-T path...")
        rnnt_acc, rnnt_loss_val = run_rnnt_overfit(
            words, n_per_word=20, num_steps=3000, lr=1e-3,
            batch_size=16, device_str=device,
        )
    else:
        print(f"\n\nCTC accuracy too low ({ctc_acc:.1f}%) — skipping RNN-T test.")
        print("This is expected on M3 Pro with small data. Real training needs GPU + more data.")
