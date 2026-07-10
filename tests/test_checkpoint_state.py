import pytest
import torch

from src.encoding.config import get_han_codec, get_legacy_han_codec
from src.training.checkpoint import normalize_model_state_keys
from src.training.taxonomy_checkpoint import migrate_taxonomy_state
from src.taxonomy import GROUPS, GROUP_SCRIPTS


def test_normalize_model_state_keys_strips_compile_wrapper_prefix():
    tensor = torch.ones(2)
    normalized, changed = normalize_model_state_keys({"_orig_mod.layer.weight": tensor})
    assert changed
    assert list(normalized) == ["layer.weight"]
    assert normalized["layer.weight"] is tensor


def test_normalize_model_state_keys_leaves_plain_state_unchanged():
    state = {"layer.weight": torch.ones(2)}
    normalized, changed = normalize_model_state_keys(state)
    assert not changed
    assert normalized is state


def test_taxonomy_migration_skips_removed_emoji_slots():
    old_names = [
        ["latin"], ["cyrillic", "greek"], ["arabic"], ["hebrew"],
        ["han"], ["kana"], ["korean"],
        ["devanagari", "gurmukhi", "gujarati", "bengali", "odia"],
        ["kannada", "telugu", "sinhala"], ["malayalam", "tamil"],
        ["thai", "lao", "burmese", "khmer"], ["emoji"],
        ["armenian", "georgian"], ["ethiopic"], ["tibetan"],
    ]
    new_names = [list(GROUP_SCRIPTS[group]) for group in GROUPS]
    old_group_rows = torch.arange(32, dtype=torch.float32).reshape(16, 2)
    state = {
        "group_head.2.weight": old_group_rows,
        "group_head.2.bias": torch.arange(16, dtype=torch.float32),
        "group_layers.0.routed_mlps.11.fc1.weight": torch.full((2, 2), 11.0),
        "group_layers.0.routed_mlps.12.fc1.weight": torch.full((2, 2), 12.0),
        "script_layers.0.routed_mlps.22.fc1.weight": torch.full((2, 2), 22.0),
        "script_layers.0.routed_mlps.23.fc1.weight": torch.full((2, 2), 23.0),
        "ctc_modules.11.heads.0.proj.weight": torch.full((3, 2), 11.0),
        "ctc_modules.12.heads.0.proj.weight": torch.full((3, 2), 12.0),
        "lid2_heads.12.2.weight": torch.full((2, 2), 12.0),
    }
    current = {
        "group_head.2.weight": torch.zeros(15, 2),
        "group_head.2.bias": torch.zeros(15),
        "group_layers.0.routed_mlps.11.fc1.weight": torch.zeros(2, 2),
        "script_layers.0.routed_mlps.22.fc1.weight": torch.zeros(2, 2),
        "ctc_modules.11.heads.0.proj.weight": torch.zeros(3, 2),
        "lid2_heads.11.2.weight": torch.zeros(2, 2),
    }

    changed = migrate_taxonomy_state(
        state,
        current,
        {"group_script_names": old_names},
        {"group_script_names": new_names},
    )

    assert changed == 6
    assert torch.equal(state["group_head.2.weight"][11], old_group_rows[12])
    assert torch.equal(state["group_head.2.weight"][14], old_group_rows[15])
    assert state["group_layers.0.routed_mlps.11.fc1.weight"].eq(12).all()
    assert state["script_layers.0.routed_mlps.22.fc1.weight"].eq(23).all()
    assert state["ctc_modules.11.heads.0.proj.weight"].eq(12).all()
    assert state["lid2_heads.11.2.weight"].eq(12).all()


def test_post_han_split_experts_shift_past_removed_emoji():
    old_names = [
        ["latin"], ["cyrillic", "greek"], ["arabic"], ["hebrew"],
        ["han_sparse", "han_dense"], ["kana"], ["korean"],
        ["devanagari", "gurmukhi", "gujarati", "bengali", "odia"],
        ["kannada", "telugu", "sinhala"], ["malayalam", "tamil"],
        ["thai", "lao", "burmese", "khmer"], ["emoji"],
        ["armenian", "georgian"], ["ethiopic"], ["tibetan"],
    ]
    new_names = [list(GROUP_SCRIPTS[group]) for group in GROUPS]
    state = {
        "script_layers.0.routed_mlps.23.fc1.weight": torch.full((2, 2), 23.0),
        "script_layers.0.routed_mlps.27.fc1.weight": torch.full((2, 2), 27.0),
    }
    current = {
        "script_layers.0.routed_mlps.22.fc1.weight": torch.zeros(2, 2),
        "script_layers.0.routed_mlps.26.fc1.weight": torch.zeros(2, 2),
    }

    changed = migrate_taxonomy_state(
        state,
        current,
        {"group_script_names": old_names},
        {"group_script_names": new_names},
    )

    assert changed == 2
    assert state["script_layers.0.routed_mlps.22.fc1.weight"].eq(23).all()
    assert state["script_layers.0.routed_mlps.26.fc1.weight"].eq(27).all()


def test_layout_uses_stable_ids_for_current_era_checkpoints():
    """A current-era checkpoint used as the OLD side of a future migration
    must resolve script experts at their stable global IDs (han_dense=26),
    not group/local flatten order (which would put han_dense at 6)."""
    from src.taxonomy import SCRIPT_TO_ID
    from src.training.taxonomy_checkpoint import _layout

    current_names = [list(GROUP_SCRIPTS[group]) for group in GROUPS]
    _, _, flat_ids = _layout(current_names)
    assert flat_ids == SCRIPT_TO_ID


@pytest.mark.parametrize("local_id,script", [(0, "han_sparse"), (1, "han_dense")])
def test_han_ctc_head_rows_follow_token_identity(local_id, script):
    """Both split Han heads must be seeded row-by-row from the legacy head,
    matched by token string — not by position — with blank and ALT slots
    carried over. Uses the real codecs built from training_data/corpora."""
    old_names = [
        ["latin"], ["cyrillic", "greek"], ["arabic"], ["hebrew"],
        ["han"], ["kana"], ["korean"],
        ["devanagari", "gurmukhi", "gujarati", "bengali", "odia"],
        ["kannada", "telugu", "sinhala"], ["malayalam", "tamil"],
        ["thai", "lao", "burmese", "khmer"], ["emoji"],
        ["armenian", "georgian"], ["ethiopic"], ["tibetan"],
    ]
    new_names = [list(GROUP_SCRIPTS[group]) for group in GROUPS]
    legacy, split = get_legacy_han_codec(), get_han_codec(script)

    # Row r of the legacy head carries the value r+1 so every source row is
    # identifiable and nonzero; unseeded target rows keep the -1 sentinel.
    old_weight = torch.arange(1, legacy.vocab_size + 1,
                              dtype=torch.float32).unsqueeze(1).expand(-1, 2)
    key = f"ctc_modules.4.heads.{local_id}.proj.weight"
    state = {"ctc_modules.4.heads.0.proj.weight": old_weight.clone()}
    current = {key: torch.full((split.vocab_size, 2), -1.0)}

    changed = migrate_taxonomy_state(
        state, current,
        {"group_script_names": old_names},
        {"group_script_names": new_names},
    )

    assert changed == 1
    result = state[key]
    assert (result != -1).all(), "every split-head row must be warm-started"
    legacy_id = {token: i + 1 for i, token in enumerate(legacy.tokens)}
    expected = [0] + [legacy_id[token] for token in split.tokens]
    expected += list(range(len(legacy.tokens) + 1, legacy.vocab_size))
    assert torch.equal(result[:, 0], old_weight[expected, 0])
