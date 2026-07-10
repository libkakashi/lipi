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
