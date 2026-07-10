import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train.train import _optimizer_group_sizes, resume_from_checkpoint


def test_optimizer_layout_ignores_lazy_state_entries():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.Linear(2, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters())

    # Only the second layer participates, so Adam has state for two of the
    # four parameters while the saved parameter-group layout remains complete.
    model[1](torch.ones(1, 2)).sum().backward()
    optimizer.step()
    state = optimizer.state_dict()

    assert len(state["state"]) == 2
    assert _optimizer_group_sizes(state) == (4,)
    assert _optimizer_group_sizes(optimizer) == (4,)


def test_optimizer_layout_detects_changed_groups():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.Linear(2, 1),
    )
    split_optimizer = torch.optim.AdamW([
        {"params": model[0].parameters()},
        {"params": model[1].parameters()},
    ])
    flat_optimizer = torch.optim.AdamW(model.parameters())

    assert _optimizer_group_sizes(split_optimizer.state_dict()) == (2, 2)
    assert _optimizer_group_sizes(flat_optimizer) == (4,)


def test_resume_loads_sparse_adam_state(tmp_path):
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.Linear(2, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler(enabled=False)

    model[1](torch.ones(1, 2)).sum().backward()
    optimizer.step()
    saved_state = optimizer.state_dict()
    checkpoint = tmp_path / "epoch1.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": saved_state,
        "scaler": scaler.state_dict(),
        "epoch": 1,
    }, checkpoint)

    resumed_model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.Linear(2, 1),
    )
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        resumed_optimizer, T_max=8)
    args = SimpleNamespace(
        resume=str(checkpoint),
        skip_backbone_load=False,
        freeze_except=None,
        lr=3e-4,
        epochs=4,
    )

    start_epoch, _, _ = resume_from_checkpoint(
        args, resumed_model, resumed_optimizer, resumed_optimizer,
        torch.amp.GradScaler(enabled=False), scheduler,
        steps_per_epoch=2, device_type="cpu")

    assert start_epoch == 2
    assert len(saved_state["state"]) == 2
    assert len(resumed_optimizer.state) == 2


def test_emoji_era_checkpoint_resumes_cleanly_into_v4(tmp_path, capsys):
    """End-to-end rehearsal of the real resume: an emoji-era checkpoint
    (15 groups incl emoji, unified han, pre-RTL objective) loaded through
    resume_from_checkpoint into a current-taxonomy model. Every surviving
    module must transfer (no missing, no shape-skipped layers), han's CTC
    head bit-identically, and optimizer + EMA must reset."""
    from src.model.encoder import LipiMoEEncoder
    from src.taxonomy import GROUPS, GROUP_SCRIPTS, SCRIPTS

    sizes = {name: 24 + 2 * i for i, name in enumerate(SCRIPTS)}
    sizes["emoji"] = 30

    old_groups = GROUPS[:11] + ["emoji"] + GROUPS[11:]
    old_names = [["emoji"] if g == "emoji" else list(GROUP_SCRIPTS[g])
                 for g in old_groups]
    new_names = [list(GROUP_SCRIPTS[g]) for g in GROUPS]

    def build(names):
        torch.manual_seed(11)
        return LipiMoEEncoder(
            dim=128, num_groups=len(names),
            group_script_vocab_sizes=[[sizes[s] for s in group]
                                      for group in names],
            group_script_names=names, drop_path_rate=0.0)

    old_model = build(old_names)
    checkpoint = tmp_path / "moe_epoch3.pt"
    old_state = {k: v.clone() for k, v in old_model.state_dict().items()}
    torch.save({
        "model": old_state,
        "model_config": old_model.config,
        "optimizer": torch.optim.AdamW(old_model.parameters()).state_dict(),
        "scaler": {},
        "ema": {"decay": 0.999},
        "epoch": 3,
        "ctc_direction_version": 1,   # pre-RTL objective
    }, checkpoint)

    model = build(new_names)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    args = SimpleNamespace(resume=str(checkpoint), skip_backbone_load=False,
                           freeze_except=None, lr=3e-4, epochs=40)

    start_epoch, _, ema_state = resume_from_checkpoint(
        args, model, optimizer, optimizer,
        torch.amp.GradScaler(enabled=False), scheduler,
        steps_per_epoch=2, device_type="cpu")
    out = capsys.readouterr().out

    assert start_epoch == 4
    assert ema_state is None
    assert not optimizer.state
    assert "Taxonomy upgraded" in out
    assert "RTL CTC objective upgraded" in out
    # A clean warm start: every current module has an old-name source.
    assert "missing from checkpoint" not in out
    assert "Skipped" not in out or "Skipping optimizer" in out
    assert "shape-mismatched" not in out

    loaded = model.state_dict()
    # han (group 4) CTC head transfers fully intact — same codec, same rows.
    for suffix in ("proj.weight", "proj.bias"):
        assert torch.equal(loaded[f"ctc_modules.4.heads.0.{suffix}"],
                           old_state[f"ctc_modules.4.heads.0.{suffix}"])
    # kana keeps flat id 6; armenian shifts 23 -> 22 past the emoji slot.
    assert torch.equal(loaded["script_layers.0.routed_mlps.6.fc1.weight"],
                       old_state["script_layers.0.routed_mlps.6.fc1.weight"])
    assert torch.equal(loaded["script_layers.0.routed_mlps.22.fc1.weight"],
                       old_state["script_layers.0.routed_mlps.23.fc1.weight"])
    # caucasus group modules shift 12 -> 11; LID-1 blank row follows.
    assert torch.equal(loaded["group_layers.0.routed_mlps.11.fc1.weight"],
                       old_state["group_layers.0.routed_mlps.12.fc1.weight"])
    assert torch.equal(loaded["group_head.2.weight"][14],
                       old_state["group_head.2.weight"][15])
    # The trunk is untouched by migration.
    assert torch.equal(loaded["swa_d.0.attn.qkv.weight"],
                       old_state["swa_d.0.attn.qkv.weight"])
