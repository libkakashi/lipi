"""
Render all CJK chars, compute visual similarity via CNN features, assign MRG slots.

Uses ResNet18 intermediate features for structural similarity instead of raw pixels.
Renders at 64x64 with multiple fonts for robustness.
"""
import numpy as np
from PIL import ImageFont, Image, ImageDraw
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import time

import torch
import torchvision.models as models
import torchvision.transforms as T

FONTS = [
    ImageFont.truetype('/System/Library/Fonts/STHeiti Medium.ttc', 52),
    ImageFont.truetype('/System/Library/Fonts/STHeiti Light.ttc', 52),
    ImageFont.truetype('/Library/Fonts/Arial Unicode.ttf', 52),
]
SIZE = 64


def render_char_single(char, font):
    img = Image.new('L', (SIZE, SIZE), 255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (SIZE - w) // 2 - bbox[0]
    y = (SIZE - h) // 2 - bbox[1]
    draw.text((x, y), char, font=font, fill=0)
    return img


def render_char_multifont(char):
    """Render with multiple fonts, return list of PIL images."""
    return [render_char_single(char, f) for f in FONTS]


def main():
    t0 = time.time()

    # All CJK chars
    all_cjk = [chr(cp) for cp in range(0x4E00, 0xA000)] + \
              [chr(cp) for cp in range(0x3400, 0x4DC0)]
    print(f"Total CJK chars: {len(all_cjk):,}")

    # Load corpus freq
    char_freq = Counter()
    for line in Path("training_data/corpora/cjk_char_freq.tsv").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            char_freq[parts[0]] = int(parts[1])

    # Rank by raw corpus frequency, skip base chars
    import sys; sys.path.insert(0, ".")
    from src.encoding.config import _CJK_BASE, _CJK_N_ALT, _CJK_VOCAB_SIZE
    base_set = set(_CJK_BASE)
    n_freq = _CJK_VOCAB_SIZE - 1 - len(_CJK_BASE) - _CJK_N_ALT  # BLANK + base + ALT

    vocab_chars = [c for c, _ in char_freq.most_common() if c not in base_set][:n_freq]

    vocab_set = set(vocab_chars) | base_set
    need_chars = [c for c in all_cjk if c not in vocab_set]
    print(f"Vocab (freq): {len(vocab_chars):,}, Base: {len(_CJK_BASE)}, Need coverage: {len(need_chars):,}")

    # Render all chars with multiple fonts
    print(f"Rendering {len(all_cjk):,} chars × {len(FONTS)} fonts...", flush=True)
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=16) as pool:
        all_renders = list(pool.map(render_char_multifont, all_cjk, chunksize=200))
    print(f"Rendered in {time.time()-t1:.1f}s")

    # Extract CNN features using ResNet18
    print("Loading ResNet18 feature extractor...", flush=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Remove final FC layer — use avgpool output (512-dim)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet = resnet.to(device).eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def pil_to_rgb_tensor(img):
        """Convert grayscale PIL to 3-channel tensor for ResNet."""
        rgb = Image.merge("RGB", [img, img, img])
        return transform(rgb)

    print(f"Extracting CNN features on {device}...", flush=True)
    t2 = time.time()
    BATCH = 128
    all_features = []

    for start in range(0, len(all_cjk), BATCH):
        end = min(start + BATCH, len(all_cjk))
        batch_tensors = []
        for i in range(start, end):
            # Average features across fonts
            font_tensors = [pil_to_rgb_tensor(img) for img in all_renders[i]]
            batch_tensors.extend(font_tensors)

        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            feats = resnet(batch).squeeze(-1).squeeze(-1).cpu().numpy()

        # Average features across fonts for each char
        n_fonts = len(FONTS)
        for i in range(end - start):
            font_feats = feats[i * n_fonts:(i + 1) * n_fonts]
            all_features.append(font_feats.mean(axis=0))

        done = end
        if done % 5000 < BATCH:
            print(f"  {done:,}/{len(all_cjk):,}", flush=True)

    print(f"Features extracted in {time.time()-t2:.1f}s")

    char_to_idx = {c: i for i, c in enumerate(all_cjk)}
    all_matrix = np.stack(all_features)  # (27k, 512)

    # Normalize for cosine similarity
    norms = np.linalg.norm(all_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    all_normed = all_matrix / norms

    # All leaf candidates = base CJK chars (radicals) + freq vocab chars
    # Only include chars that were rendered (i.e. in all_cjk)
    leaf_seen = set()
    leaf_chars = []
    for c in _CJK_BASE:
        if c in char_to_idx and c not in leaf_seen:
            leaf_chars.append(c)
            leaf_seen.add(c)
    for c in vocab_chars:
        if c in char_to_idx and c not in leaf_seen:
            leaf_chars.append(c)
            leaf_seen.add(c)
    leaf_indices = [char_to_idx[c] for c in leaf_chars]
    need_indices = [char_to_idx[c] for c in need_chars if c in char_to_idx]

    vocab_matrix = all_normed[leaf_indices]
    need_matrix = all_normed[need_indices]

    print(f"Computing similarity ({len(need_indices):,} × {len(leaf_indices):,})...", flush=True)
    t3 = time.time()

    CHUNK = 2000
    n_vocab = len(leaf_indices)
    similarity_topk = []

    for start in range(0, len(need_indices), CHUNK):
        end = min(start + CHUNK, len(need_indices))
        chunk = need_matrix[start:end]
        sims = chunk @ vocab_matrix.T

        # Full sort — we need all candidates to guarantee zero skips
        sorted_indices = np.argsort(-sims, axis=1)
        for i in range(end - start):
            row_sorted = sorted_indices[i]
            row_scores = sims[i, row_sorted]
            similarity_topk.append([(int(row_sorted[j]), float(row_scores[j])) for j in range(n_vocab)])

        done = end
        if done % 5000 < CHUNK:
            print(f"  {done:,}/{len(need_indices):,}", flush=True)

    print(f"Similarity computed in {time.time()-t3:.1f}s")

    # Sort by freq (highest first gets best match)
    need_with_freq = sorted(
        range(len(need_chars)),
        key=lambda i: char_freq.get(need_chars[i], 0),
        reverse=True,
    )

    total_occ = sum(char_freq.values())
    single_occ = sum(char_freq.get(c, 0) for c in vocab_set)

    MAX_SLOTS = 24
    slots_used = Counter()
    assigned = {}
    skipped = []
    rank_dist = Counter()

    for idx in need_with_freq:
        char = need_chars[idx]
        if idx >= len(similarity_topk):
            skipped.append((char, char_freq.get(char, 0)))
            continue
        topk = similarity_topk[idx]
        placed = False
        for rank, (vocab_local_idx, score) in enumerate(topk):
            if slots_used[vocab_local_idx] < MAX_SLOTS:
                slot = slots_used[vocab_local_idx]
                assigned[char] = (slot, leaf_chars[vocab_local_idx], score)
                slots_used[vocab_local_idx] += 1
                rank_dist[rank] += 1
                placed = True
                break
        if not placed:
            skipped.append((char, char_freq.get(char, 0)))

    assigned_occ = sum(char_freq.get(c, 0) for c in assigned)
    avg_score = np.mean([s for _, _, s in assigned.values()]) if assigned else 0
    got_1st = rank_dist.get(0, 0)
    got_top3 = sum(rank_dist.get(i, 0) for i in range(3))
    print(f"\nTop 20 assigned (highest freq):")
    top = sorted(assigned.items(), key=lambda x: -char_freq.get(x[0], 0))[:20]
    for char, (slot, match, score) in top:
        print(f"  {char} (freq={char_freq.get(char,0):,}) → ALT{slot:02d} + {match} (sim={score:.3f})")

    if skipped:
        print(f"\nTop 10 skipped:")
        for char, freq in sorted(skipped, key=lambda x: -x[1])[:10]:
            print(f"  {char} freq={freq}")

    # Save mapping
    out = Path("training_data/corpora/cjk_visual_mapping.tsv")
    with open(out, "w") as f:
        f.write("char\tslot\tmatch\tscore\n")
        for char, (slot, match, score) in sorted(
                assigned.items(), key=lambda x: -char_freq.get(x[0], 0)):
            f.write(f"{char}\t{slot}\t{match}\t{score:.4f}\n")
    print(f"\nSaved to {out}")

    # Save the exact vocab list (single-token chars, in order)
    vocab_out = Path("training_data/corpora/cjk_vocab.txt")
    with open(vocab_out, "w", encoding="utf-8") as f:
        for char in vocab_chars:
            f.write(f"{char}\n")
    print(f"Saved vocab to {vocab_out} ({len(vocab_chars)} chars)")

    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
