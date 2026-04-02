#!/usr/bin/env python3
"""Quick diagnostic: checks data labels, gradient norms, and LID-1 signal."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model.lid import SCRIPT_TO_ID, GROUP_TO_ID, SCRIPT_TO_GROUP, GROUPS, SCRIPTS

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/all_shards_v5")

print("=" * 60)
print("1. CHECKING DATA LABELS vs lid.py")
print("=" * 60)

shard_files = sorted(data_dir.glob("shard_*.pt")) + sorted(data_dir.glob("char_shard_*.pt"))
print(f"Found {len(shard_files)} shards in {data_dir}")

# Sample a few shards and check IDs
bad_shards = []
gid_counts = {}
sid_counts = {}
total_samples = 0

for sf in shard_files[:10]:  # check first 10 shards
    s = torch.load(sf, weights_only=False)
    gids = s["group_ids"]
    sids = s["script_ids"]
    labels = s["labels"]
    n = len(labels)
    total_samples += n

    for gid in gids.unique().tolist():
        gid_counts[gid] = gid_counts.get(gid, 0) + (gids == gid).sum().item()
    for sid in sids.unique().tolist():
        sid_counts[sid] = sid_counts.get(sid, 0) + (sids == sid).sum().item()

    # Verify consistency: script_id should map to group_id correctly
    for sid_val in sids.unique().tolist():
        if sid_val >= len(SCRIPTS):
            bad_shards.append((sf.name, f"script_id {sid_val} out of range (max {len(SCRIPTS)-1})"))
            continue
        script_name = SCRIPTS[sid_val]
        expected_gid = GROUP_TO_ID[SCRIPT_TO_GROUP[script_name]]
        actual_gids = gids[sids == sid_val].unique().tolist()
        if actual_gids != [expected_gid]:
            bad_shards.append((sf.name, f"script {script_name} (sid={sid_val}): "
                             f"group_id={actual_gids}, expected={expected_gid} ({SCRIPT_TO_GROUP[script_name]})"))

print(f"\nSampled {total_samples} samples from {min(10, len(shard_files))} shards")

print(f"\nGroup ID distribution:")
for gid in sorted(gid_counts):
    gname = GROUPS[gid] if gid < len(GROUPS) else f"UNKNOWN({gid})"
    print(f"  {gid:>2} {gname:<20} {gid_counts[gid]:>6} samples")

print(f"\nScript ID distribution:")
for sid in sorted(sid_counts):
    sname = SCRIPTS[sid] if sid < len(SCRIPTS) else f"UNKNOWN({sid})"
    print(f"  {sid:>2} {sname:<20} {sid_counts[sid]:>6} samples")

if bad_shards:
    print(f"\n*** FOUND {len(bad_shards)} LABEL MISMATCHES ***")
    for shard, msg in bad_shards:
        print(f"  {shard}: {msg}")
else:
    print(f"\n✓ All script→group mappings are consistent with lid.py")

# Check metadata too
meta_path = data_dir / "metadata.pt"
if meta_path.exists():
    meta = torch.load(meta_path, weights_only=False)
    print(f"\nMetadata scripts: {meta.get('active_scripts', 'MISSING')}")
    print(f"Metadata groups:  {meta.get('active_groups', 'MISSING')}")
    # Check if metadata groups match lid.py
    meta_groups = meta.get('active_groups', [])
    if meta_groups != list(GROUPS[:len(meta_groups)]):
        print(f"*** METADATA GROUP ORDER MISMATCH ***")
        print(f"  metadata: {meta_groups}")
        print(f"  lid.py:   {list(GROUPS[:len(meta_groups)])}")

print(f"\n{'=' * 60}")
print("2. CHECKING GRADIENT FLOW (one batch)")
print("=" * 60)

# Quick gradient check with synthetic data
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")

from src.model.moe_encoder import LipiMoEEncoder
from src.data.vocab import get_all_script_vocabs

active_scripts = list(SCRIPTS)
active_groups = list(GROUPS)
_, group_vocab_sizes = get_all_script_vocabs(active_scripts, active_groups)
group_script_names = []
for g, gname in enumerate(active_groups):
    scripts = [s for s in active_scripts if SCRIPT_TO_GROUP.get(s) == gname]
    group_script_names.append(scripts)

model = LipiMoEEncoder(
    num_groups=len(active_groups),
    group_script_vocab_sizes=group_vocab_sizes,
    group_script_names=group_script_names,
).to(device)

# Synthetic batch
B = 16
imgs = torch.randn(B, 2, 32, 64, device=device)
gids = torch.randint(0, len(active_groups), (B,), device=device)

model.train()
with torch.amp.autocast(device.type, enabled=(device.type == "cuda"), dtype=torch.bfloat16):
    out = model(imgs, group_ids=None)

loss = torch.nn.functional.cross_entropy(out["group_logits"], gids)
(3 * loss).backward()  # lid1_weight=3

# Measure gradient norms per component
def grad_norm(params):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return torch.cat([g.flatten() for g in grads]).norm().item()

lid1_params = [p for n, p in model.named_parameters() if "lid_coarse" in n]
shared_params = [p for n, p in model.named_parameters()
                 if not any(k in n for k in ("stage1.", "stage2.", "ctc_modules.", "lid_coarse"))]
expert_params = [p for n, p in model.named_parameters()
                 if any(k in n for k in ("stage1.", "stage2.", "ctc_modules."))]

lid1_norm = grad_norm(lid1_params)
shared_norm = grad_norm(shared_params)
expert_norm = grad_norm(expert_params)
total_norm = grad_norm(list(model.parameters()))

print(f"Gradient norms (LID-1 loss only, weight=3):")
print(f"  LID-1 classifier:  {lid1_norm:.2f}  ({sum(p.numel() for p in lid1_params)/1e3:.0f}K params)")
print(f"  Shared SWA + stem: {shared_norm:.2f}  ({sum(p.numel() for p in shared_params)/1e6:.1f}M params)")
print(f"  Expert + CTC:      {expert_norm:.2f}  ({sum(p.numel() for p in expert_params)/1e6:.1f}M params)")
print(f"  Total:             {total_norm:.2f}")
print(f"  max_norm=25 clip:  {25.0/total_norm:.1%} of gradient preserved" if total_norm > 25 else "  max_norm=25: no clipping needed")
print(f"  LID-1 share:       {lid1_norm/total_norm:.1%} of total norm" if total_norm > 0 else "")

if total_norm > 25:
    effective_lid1 = lid1_norm * (25.0 / total_norm)
    print(f"\n  *** LID-1 effective gradient after clip: {effective_lid1:.4f} (was {lid1_norm:.2f})")
    print(f"  *** This means LID-1 gets {effective_lid1/lid1_norm:.1%} of its gradient — rest is wasted on expert blocks")

print(f"\n{'=' * 60}")
print("DONE")
print("=" * 60)
