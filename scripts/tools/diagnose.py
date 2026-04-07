"""
Diagnose CJK training issues. Run on the training machine:
    python -m scripts.diagnose_training --data data/shards_cjk
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1. Load data ──
    print("\n" + "=" * 60)
    print("1. LOADING DATA")
    print("=" * 60)

    from src.training.dataloader import (
        load_shards, build_script_tokenizers, encode_labels, remap_ids,
    )
    from src.model.lid import SCRIPT_TO_GROUP

    images, labels, sids_global, gids_global, meta, _, _ = load_shards(Path(args.data))
    active_scripts = meta["active_scripts"]
    active_groups = list(dict.fromkeys(
        SCRIPT_TO_GROUP[s] for s in active_scripts if s in SCRIPT_TO_GROUP))
    n_groups = len(active_groups)

    print(f"  Samples: {len(labels)}")
    print(f"  Images: {images.shape}")
    print(f"  Scripts: {active_scripts}")
    print(f"  Groups: {active_groups}")

    group_ids, local_sids, _ = remap_ids(
        active_scripts, active_groups, sids_global, gids_global)

    group_toks, group_vocab_sizes, group_names = build_script_tokenizers(
        active_scripts, active_groups)
    print(f"  Vocab sizes: {group_vocab_sizes}")

    # ── 2. Encode labels ──
    print("\n" + "=" * 60)
    print("2. ENCODING LABELS")
    print("=" * 60)

    targets, tgt_lens = encode_labels(
        labels, group_ids, local_sids, active_groups, group_toks)

    enc_len = images.shape[3] // 4
    print(f"  Encoder length T = {enc_len}")
    print(f"  Target lengths: min={tgt_lens.min()}, max={tgt_lens.max()}, "
          f"mean={tgt_lens.float().mean():.1f}")
    print(f"  Zero-length: {(tgt_lens == 0).sum()}")
    too_long = tgt_lens > enc_len
    print(f"  Too long (> {enc_len}): {too_long.sum()}")
    valid = (tgt_lens > 0) & (tgt_lens <= enc_len)
    print(f"  Valid: {valid.sum()} / {len(valid)}")

    # Check for blank IDs in targets
    blank_count = sum(
        1 for i in range(len(targets))
        if tgt_lens[i] > 0 and (targets[i, :tgt_lens[i]] == 0).any())
    print(f"  Samples with blank ID in targets: {blank_count}")

    # ── 3. Verify decomposition ──
    print("\n" + "=" * 60)
    print("3. DECOMPOSITION CHECK")
    print("=" * 60)

    from src.encoding.decompose import (
        decompose_han_kana, reconstruct_han_kana,
        _load_arbitrary_encoding,
    )
    enc = _load_arbitrary_encoding("han_kana")

    if enc["char_to_tokens"]:
        print(f"  Decomposition table loaded: {len(enc['char_to_tokens'])} chars")
    else:
        print("  *** DECOMPOSITION TABLE EMPTY — THIS IS THE BUG ***")
        print("  Check that training_data/word_lists/cjk_char_codes.tsv exists")
        sys.exit(1)

    # Check roundtrip on actual labels
    rt_fail = 0
    for i in range(min(1000, len(labels))):
        label = labels[i]
        dec = decompose_han_kana(label)
        rec = reconstruct_han_kana(list(dec))
        if rec != label:
            rt_fail += 1
            if rt_fail <= 3:
                print(f"  ROUNDTRIP FAIL: '{label}' -> '{rec}'")
    print(f"  Roundtrip failures: {rt_fail}/1000")

    # ── 4. Image check ──
    print("\n" + "=" * 60)
    print("4. IMAGE CHECK")
    print("=" * 60)

    print(f"  Shape: {images.shape}, dtype: {images.dtype}")
    print(f"  Min: {images.min():.4f}, Max: {images.max():.4f}, "
          f"Mean: {images.mean():.4f}")

    ink = images.abs().sum(dim=(1, 2, 3))
    blank = (ink < 10).sum()
    print(f"  Blank images (< 10 ink): {blank} / {len(images)}")

    # ── 5. Sample inspection ──
    print("\n" + "=" * 60)
    print("5. SAMPLE INSPECTION")
    print("=" * 60)

    for i in range(min(10, len(labels))):
        label = labels[i]
        dec = decompose_han_kana(label)
        ids = targets[i, :tgt_lens[i]].tolist()
        img_ink = images[i].abs().sum().item()
        print(f"  [{i}] label='{label}' tgt_len={tgt_lens[i]} "
              f"ids={ids[:8]}{'...' if len(ids) > 8 else ''} ink={img_ink:.0f}")

    # ── 6. Mini training loop ──
    print("\n" + "=" * 60)
    print("6. MINI TRAINING (100 steps, batch=32, fresh model)")
    print("=" * 60)

    from src.model.encoder import LipiMoEEncoder

    model = LipiMoEEncoder(
        shared_dim=256, shared_blocks_4x4=4, shared_blocks_4x16=2,
        stage1_dim=256, stage1_blocks=6,
        stage2_dim=256, stage2_blocks=4,
        num_groups=n_groups,
        group_script_vocab_sizes=group_vocab_sizes,
        group_script_names=group_names,
        head_hidden=384,
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Use only valid samples
    keep = valid.nonzero(as_tuple=True)[0]
    v_images = images[keep]
    v_targets = targets[keep]
    v_tgt_lens = tgt_lens[keep]
    v_gids = group_ids[keep]
    v_sids = local_sids[keep]

    losses = []
    bs = min(32, len(v_images))
    steps = min(100, len(v_images) // bs)

    for step in range(steps):
        s, e = step * bs, (step + 1) * bs
        imgs = v_images[s:e].to(device)
        tgts = v_targets[s:e].to(device)
        tlens = v_tgt_lens[s:e].to(device)
        gids_b = v_gids[s:e].to(device)
        sids_b = v_sids[s:e].to(device)

        out = model(imgs, group_ids=gids_b, script_ids=sids_b)

        # Compute CTC loss directly
        ok = (tlens <= out["lengths"]) & (tlens > 0)
        if not ok.any():
            print(f"  Step {step}: NO VALID SAMPLES (all targets > enc_length)")
            continue

        ctc_total = torch.zeros(1, device=device)
        ctc_n = 0
        for g in range(n_groups):
            for s_idx, vs in enumerate(group_vocab_sizes[g]):
                mask = ok & (gids_b == g) & (sids_b == s_idx)
                if not mask.any():
                    continue
                log_probs = (out["logits"][mask, :, :vs].float()
                             .log_softmax(dim=-1).permute(1, 0, 2))
                loss = F.ctc_loss(
                    log_probs, tgts[mask], out["lengths"][mask], tlens[mask],
                    blank=0, reduction="sum", zero_infinity=True)
                ctc_total += loss
                ctc_n += tlens[mask].sum().item()

        if ctc_n > 0:
            ctc_loss = ctc_total / ctc_n
        else:
            ctc_loss = torch.zeros(1, device=device, requires_grad=True)

        optimizer.zero_grad()
        ctc_loss.backward()

        # Check gradient norms
        shared_norm = torch.nn.utils.clip_grad_norm_(
            [p for n, p in model.named_parameters()
             if not any(k in n for k in ("stage1.", "stage2.", "ctc_modules."))],
            max_norm=25.0)
        expert_norm = torch.nn.utils.clip_grad_norm_(
            [p for n, p in model.named_parameters()
             if any(k in n for k in ("stage1.", "stage2.", "ctc_modules."))],
            max_norm=25.0)

        optimizer.step()

        loss_val = ctc_loss.item()
        losses.append(loss_val)

        if step % 10 == 0 or step == steps - 1:
            print(f"  Step {step:3d}: ctc={loss_val:.4f}  "
                  f"gnorm s={shared_norm:.2f} e={expert_norm:.2f}")

    # ── 7. Diagnosis ──
    print("\n" + "=" * 60)
    print("7. DIAGNOSIS")
    print("=" * 60)

    if not losses:
        print("  *** NO TRAINING HAPPENED — check data ***")
    elif losses[-1] < losses[0] * 0.9:
        print(f"  Loss decreased: {losses[0]:.2f} -> {losses[-1]:.2f}")
        print("  Model CAN learn from this data.")
        print("  The issue is likely:")
        print("    - Checkpoint resume corrupted weights (try without --resume)")
        print("    - LR schedule decayed too fast (try --epochs 50 --lr 1e-3)")
    else:
        print(f"  Loss STUCK: {losses[0]:.2f} -> {losses[-1]:.2f}")
        print("  Possible causes:")
        print("    - Images don't match labels (check section 5)")
        print("    - Images are blank/unreadable (check section 4)")
        print("    - Encoding is broken (check section 3)")


if __name__ == "__main__":
    main()
