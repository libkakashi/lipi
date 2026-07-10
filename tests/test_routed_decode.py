import torch

from scripts.inference.run_doctr import _decode_one
from src.encoding.decompose import encode_text, script_vocab_size
from src.taxonomy import GROUP_SCRIPTS


def test_decode_one_joins_sparse_dense_and_kana_runs():
    scripts = [list(GROUP_SCRIPTS[group]) for group in (
        "latin", "cyrillic_greek", "arabic", "hebrew", "han", "kana")]
    routed = [
        (4, 0, "han_sparse", "山"),
        (4, 1, "han_dense", "語"),
        (5, 0, "kana", "か"),
    ]
    max_vocab = max(script_vocab_size(script) for *_, script, _ in routed)
    logits = torch.full((12, max_vocab), -10.0)
    groups = []
    local_scripts = []
    offset = 0
    for group, local_script, script, text in routed:
        token = encode_text(text, script)[0]
        for value in (0, token, token, 0):
            logits[offset, value] = 10.0
            groups.append(group)
            local_scripts.append(local_script)
            offset += 1

    group, script, text, confidence, runs = _decode_one(
        logits,
        torch.tensor(groups),
        torch.tensor(local_scripts),
        scripts,
    )

    assert group == 4
    assert script == "han_sparse"
    assert text == "山語か"
    assert confidence > 0.99
    assert [run[1] for run in runs] == ["han_sparse", "han_dense", "kana"]
