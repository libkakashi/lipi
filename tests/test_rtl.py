"""Regression tests for RTL rendering and CTC traversal."""

import torch

import src.training.eval as eval_mod
import src.training.losses as losses_mod
from scripts.train.train import resume_from_checkpoint
from src.data.rendering import visual_order_blocks
from src.data.script_detect import split_by_script
from src.data.word_lists import _rtl_word_is_encodable
from src.encoding.direction import (
    ctc_time_order,
    is_rtl_script,
    segment_is_rtl,
    undo_visual_digit_order,
)


def _block(text: str):
    return (object(), 10, text, 0, 0, 10)


def _gap():
    return (None, 0, "", 0, 0, 4)


def _texts(blocks):
    return [block[2] for block in blocks]


def test_ctc_time_order_reverses_only_rtl_scripts():
    seq = torch.tensor([1, 2, 3])
    assert is_rtl_script("arabic")
    assert is_rtl_script("hebrew")
    assert torch.equal(ctc_time_order(seq, "arabic"), torch.tensor([3, 2, 1]))
    assert torch.equal(ctc_time_order(seq, "hebrew"), torch.tensor([3, 2, 1]))
    assert torch.equal(ctc_time_order(seq, "latin"), seq)


def test_visual_order_reverses_rtl_words_and_keeps_ltr_run():
    logical = [
        _block("שלום"), _gap(),
        _block("hello"), _gap(), _block("world"), _gap(),
        _block("עולם"),
    ]
    visual = visual_order_blocks(logical, text_index=2)
    assert [text for text in _texts(visual) if text] == [
        "עולם", "hello", "world", "שלום"]


def test_visual_order_moves_split_suffix_punctuation_to_rtl_left_edge():
    logical = [_block("مرحبا"), _block("?")]
    visual = visual_order_blocks(logical, text_index=2)
    assert _texts(visual) == ["?", "مرحبا"]


def test_segment_is_rtl_excludes_digit_runs():
    assert segment_is_rtl("arabic", "سال")
    assert segment_is_rtl("hebrew", "שלום")
    # Digit runs render left-to-right even inside RTL text.
    assert not segment_is_rtl("arabic", "۱۳۹۸")
    assert not segment_is_rtl("arabic", "١٢٣")
    assert not segment_is_rtl("latin", "hello")


def test_split_by_script_peels_digit_runs_from_rtl_words():
    assert split_by_script("سال۱۳۹۸", "arabic") == [
        ("سال", "arabic"), ("۱۳۹۸", "arabic")]
    assert split_by_script("۱۲ماه۳۴", "arabic") == [
        ("۱۲", "arabic"), ("ماه", "arabic"), ("۳۴", "arabic")]
    # ASCII digits keep their existing latin peel-off.
    assert split_by_script("שנת2024", "hebrew") == [
        ("שנת", "hebrew"), ("2024", "latin")]
    # Non-RTL parents keep their digits attached.
    assert split_by_script("वर्ष२०", "devanagari") == [("वर्ष२०", "devanagari")]


def test_undo_visual_digit_order_restores_digit_runs():
    assert undo_visual_digit_order("۸۹۳۱") == "۱۳۹۸"
    assert undo_visual_digit_order("لاس ۸۹۳۱ رخآ") == "لاس ۱۳۹۸ رخآ"
    assert undo_visual_digit_order("بدون رقم") == "بدون رقم"
    assert undo_visual_digit_order("") == ""


def test_rtl_digit_segments_use_frame_order(monkeypatch):
    """An RTL-script digit segment must consume frames left-to-right."""
    monkeypatch.setattr(losses_mod, "_encode_text_cached",
                        lambda text, script: (1, 2))
    logits = torch.full((1, 4, 3), -10.0)
    for t, token in enumerate([0, 1, 2, 0]):  # canvas order = target order
        logits[0, t, token] = 10.0
    segments = [[{"group_id": 0, "script_id": 0, "text": "۱۲",
                  "offset": 0, "width": 16}]]
    loss = losses_mod.compute_ctc_loss_segments(
        logits, segments, torch.tensor([4]), [["arabic"]], [[3]])
    assert loss.item() < 1e-6


def test_visual_order_base_direction_skips_number_units():
    # Logical: number first, then an Arabic word — base stays RTL, so the
    # logically-first number renders at the right edge.
    logical = [_block("۲۰"), _gap(), _block("شمس")]
    visual = visual_order_blocks(logical, text_index=2)
    assert [text for text in _texts(visual) if text] == ["شمس", "۲۰"]


def test_visual_order_keeps_leading_gap():
    logical = [_gap(), _block("שלום"), _gap(), _block("עולם")]
    visual = visual_order_blocks(logical, text_index=2)
    assert len(visual) == len(logical)
    assert _texts(visual) == ["", "עולם", "", "שלום"]


def test_rtl_word_filter_rejects_silently_dropped_characters():
    assert _rtl_word_is_encodable("العربية", "arabic")
    assert _rtl_word_is_encodable("فارسی", "arabic")
    assert _rtl_word_is_encodable("שלום", "hebrew")
    assert not _rtl_word_is_encodable("ا\u0750ب", "arabic")
    assert not _rtl_word_is_encodable("\ufe8eا", "arabic")
    assert not _rtl_word_is_encodable("می\u200cشود", "arabic")


def _directional_logits():
    logits = torch.full((2, 4, 3), -10.0)
    # Physical canvas order. Arabic must traverse this sequence backwards;
    # Latin consumes it as-is. Both logical targets are token IDs [1, 2].
    for t, token in enumerate([0, 2, 1, 0]):
        logits[0, t, token] = 10.0
    for t, token in enumerate([0, 1, 2, 0]):
        logits[1, t, token] = 10.0
    segments = [
        [{"group_id": 0, "script_id": 0, "text": "اب",
          "offset": 0, "width": 16}],
        [{"group_id": 1, "script_id": 0, "text": "ab",
          "offset": 0, "width": 16}],
    ]
    return logits, segments, [["arabic"], ["latin"]], [[3], [3]]


def test_segment_ctc_loss_uses_logical_reading_direction(monkeypatch):
    monkeypatch.setattr(losses_mod, "_encode_text_cached",
                        lambda text, script: (1, 2))
    logits, segments, names, vocabs = _directional_logits()
    loss = losses_mod.compute_ctc_loss_segments(
        logits, segments, torch.tensor([4, 4]), names, vocabs)
    assert loss.item() < 1e-6


def test_validation_ctc_loss_uses_same_rtl_direction(monkeypatch):
    monkeypatch.setattr(eval_mod, "encode_text", lambda text, script: [1, 2])
    logits, segments, names, vocabs = _directional_logits()
    loss, chars = eval_mod._batched_ctc_val_loss(
        logits, segments, names, vocabs, torch.device("cpu"))
    assert chars == 4
    assert loss / chars < 1e-6


def test_legacy_lipi_checkpoint_resets_training_state(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()

    checkpoint = tmp_path / "legacy.pt"
    torch.save({
        "model": model.state_dict(),
        "model_config": {},
        "optimizer": optimizer.state_dict(),
        "scaler": {},
        "ema": {"legacy": True},
        "epoch": 3,
    }, checkpoint)

    resumed_model = torch.nn.Linear(2, 2)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        resumed_optimizer, T_max=10)
    args = type("Args", (), {
        "resume": str(checkpoint),
        "skip_backbone_load": False,
        "freeze_except": None,
        "lr": 3e-4,
        "epochs": 12,
    })()

    start_epoch, _, ema_state = resume_from_checkpoint(
        args, resumed_model, resumed_optimizer, resumed_optimizer,
        torch.amp.GradScaler(enabled=False), scheduler,
        steps_per_epoch=2, device_type="cpu")

    assert start_epoch == 4
    assert not resumed_optimizer.state
    assert ema_state is None
