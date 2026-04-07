"""
Fixed vocabulary definitions per script.

All vocabs are pre-computed and stored as hex code point files in
frozen_vocabs/. One hex code point per line (e.g., "0041" for 'A').
No runtime computation, no Unicode enumeration, no data dependency.

Each script's full vocab = [BLANK_TOKEN] + chars from file.
"""

from pathlib import Path

from src.data.tokenizer import BLANK_TOKEN

_VOCAB_DIR = Path(__file__).parent / "frozen_vocabs"


def build_script_vocab(script: str, group: str) -> list[str]:
    """Load the frozen vocab for a script.

    Returns:
        [BLANK_TOKEN, char_1, char_2, ...] — ready for LipiTokenizer.
    """
    path = _VOCAB_DIR / f"{script}_vocab.txt"
    tokens = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if line:
            tokens.append(chr(int(line, 16)))
    return [BLANK_TOKEN] + tokens


def get_all_script_vocabs(
    active_scripts: list[str],
    active_groups: list[str],
) -> tuple[list[list[str]], list[list[int]]]:
    """Load frozen vocabs for all active scripts, organized by group.

    Returns:
        group_vocabs[g][s]: token list for script s in group g
        group_vocab_sizes[g][s]: vocab size
    """
    from src.model.lid import SCRIPT_TO_GROUP

    group_vocabs = []
    group_vocab_sizes = []

    for g, group_name in enumerate(active_groups):
        scripts_in_group = [s for s in active_scripts
                            if SCRIPT_TO_GROUP.get(s) == group_name]
        vocabs = []
        sizes = []
        for script in scripts_in_group:
            vocab = build_script_vocab(script, group_name)
            vocabs.append(vocab)
            sizes.append(len(vocab))
            print(f"    Group {g} ({group_name}) / {script}: {len(vocab)} tokens")
        group_vocabs.append(vocabs)
        group_vocab_sizes.append(sizes)

    return group_vocabs, group_vocab_sizes
