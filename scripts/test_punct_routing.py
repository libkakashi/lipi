"""
Test whether LID-1 routes ASCII punctuation frames to latin.

Run on training machine with epoch 19 checkpoint:
    python scripts/test_punct_routing.py --checkpoint checkpoints/moe/moe_epoch19.pt

Tests:
  1. Pure punctuation images → does LID-1 predict latin?
  2. Mixed script+punct images → do punctuation frames route to latin?
  3. Latin CTC decode on punctuation → does it decode correctly?
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.model.encoder import LipiMoEEncoder
from src.model.lid import GROUPS, GROUP_TO_ID, SCRIPTS, SCRIPT_TO_GROUP, NUM_GROUPS
from src.data.fonts import find_fonts_for_script
from src.data.rendering import render_word, resize_or_pad
from src.encoding.decompose import encode_text, decode_ids, script_vocab_size
from src.training.dataloader import build_script_tokenizers


def img_to_tensor(img):
    """PIL Image → (1, 3, 32, W) float tensor."""
    img = img.convert("RGB")
    if img.height != 32:
        w = max(1, int(img.width * 32 / img.height))
        img = img.resize((w, 32), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def ctc_greedy_decode(logits, length):
    """Simple greedy CTC decode: collapse repeats, remove blanks."""
    preds = logits[:length].argmax(dim=-1).cpu().tolist()
    collapsed = []
    prev = -1
    for p in preds:
        if p != prev:
            if p != 0:
                collapsed.append(p)
            prev = p
    return collapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Build model
    all_scripts = list(SCRIPT_TO_GROUP.keys())
    active_groups = list(GROUPS)
    _, group_script_vocab_sizes, group_script_names = build_script_tokenizers(
        all_scripts, active_groups)

    model = LipiMoEEncoder(
        dim=256, num_groups=NUM_GROUPS,
        group_script_vocab_sizes=group_script_vocab_sizes,
        group_script_names=group_script_names,
    ).to(device).eval()

    # Load checkpoint
    print(f"Loading {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"  Missing: {len(missing)} keys")
    if unexpected:
        print(f"  Unexpected: {len(unexpected)} keys")
    del ckpt

    # Find a CJK font and a Latin font
    cjk_fonts = find_fonts_for_script("han")
    latin_fonts = find_fonts_for_script("latin")
    cjk_font = str(cjk_fonts[0]) if cjk_fonts else None
    latin_font = str(latin_fonts[0]) if latin_fonts else None
    assert cjk_font, "No CJK fonts found"
    assert latin_font, "No Latin fonts found"

    latin_gid = GROUP_TO_ID["latin"]

    # =========================================================================
    # Test 1: Pure punctuation → LID-1 routing
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Pure punctuation images — LID-1 group prediction")
    print("=" * 70)

    pure_punct = [
        ".", ",", "!", "?", ":", ";",
        "()", "[]", "{}", "\"\"",
        "123", "2024", "45.6", "$100",
        "#@&", "---", "...", "+-=",
    ]

    correct = 0
    total = 0
    for text in pure_punct:
        img = render_word(text, latin_font, clean=True)
        if img is None:
            continue
        tensor = img_to_tensor(img).to(device)
        with torch.no_grad():
            out = model(tensor, compute_ctc=False)
        # Per-frame group predictions (majority vote)
        frame_preds = out["group_logits"][0].argmax(dim=-1)  # (T,)
        # Exclude blank group
        non_blank = frame_preds[frame_preds < NUM_GROUPS]
        if len(non_blank) == 0:
            pred_group = "blank"
        else:
            pred_group = GROUPS[non_blank.mode().values.item()]
        is_latin = pred_group == "latin"
        correct += is_latin
        total += 1
        status = "OK" if is_latin else "MISS"
        print(f"  [{status}]  {text!r:10s} → {pred_group}")

    print(f"\n  Routed to latin: {correct}/{total} ({100*correct/total:.0f}%)")

    # =========================================================================
    # Test 2: Mixed script + punctuation — per-frame routing
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Mixed script+punct — per-frame LID-1 predictions")
    print("=" * 70)

    mixed_samples = [
        ("東京（2024）", "han", cjk_font),
        ("价格：$100", "han", cjk_font),
        ("日本語、中国語", "han", cjk_font),
        ("Hello, world!", "latin", latin_font),
        ("Test #123", "latin", latin_font),
    ]

    for text, expected_script, font in mixed_samples:
        img = render_word(text, font, clean=True)
        if img is None:
            print(f"  SKIP: couldn't render {text!r}")
            continue
        tensor = img_to_tensor(img).to(device)
        with torch.no_grad():
            out = model(tensor, compute_ctc=False)
        frame_preds = out["group_logits"][0].argmax(dim=-1).cpu()

        # Count groups
        from collections import Counter
        group_counts = Counter()
        for g in frame_preds.tolist():
            name = GROUPS[g] if g < NUM_GROUPS else "blank"
            group_counts[name] += 1

        total_frames = len(frame_preds)
        print(f"\n  {text!r} ({total_frames} frames):")
        for g_name, count in group_counts.most_common():
            pct = 100 * count / total_frames
            print(f"    {g_name:20s} {count:3d} frames ({pct:5.1f}%)")

    # =========================================================================
    # Test 3: Latin CTC decode on pure punctuation
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Latin CTC decode on punctuation")
    print("=" * 70)

    # Force-route everything to latin group to check CTC
    test_words = ["Hello!", "2024", "(test)", "$100", "3.14", "A+B=C"]

    correct = 0
    total = 0
    for text in test_words:
        img = render_word(text, latin_font, clean=True)
        if img is None:
            continue
        tensor = img_to_tensor(img).to(device)
        # Run with forced latin group routing
        B, _, _, W = tensor.shape
        T = W // 2
        latin_gids = torch.full((1, T), latin_gid, dtype=torch.long, device=device)
        latin_sids = torch.zeros(1, T, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(tensor, group_ids=latin_gids, script_ids=latin_sids)
        logits = out["logits"][0]  # (T, vocab)
        length = out["lengths"][0].item()
        pred_ids = ctc_greedy_decode(logits, length)
        decoded = decode_ids(pred_ids, "latin")
        match = decoded == text
        correct += match
        total += 1
        status = "OK" if match else "MISS"
        print(f"  [{status}]  {text!r:15s} → {decoded!r}")

    print(f"\n  Correct: {correct}/{total} ({100*correct/total:.0f}%)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("If Test 1 shows >80% routing to latin AND Test 3 shows >80% CTC")
    print("accuracy, it's safe to remove ASCII_COMMON from non-latin codecs")
    print("(after implementing boundary splitting in the data generator).")


if __name__ == "__main__":
    main()
