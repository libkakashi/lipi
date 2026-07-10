"""Deterministic sparse/dense routing for Han characters."""

from __future__ import annotations

import functools
from pathlib import Path


HAN_SPARSE = "han_sparse"
HAN_DENSE = "han_dense"
HAN_SCRIPTS = frozenset({HAN_SPARSE, HAN_DENSE})
HAN_SPLIT_VERSION = 1

# The direct-token split is generated from Noto Sans CJK outline complexity by
# scripts/data/build_han_split.py. IDS is the deterministic fallback for the
# small number of Han characters outside the generated tables.
_DENSE_COMPONENT_THRESHOLD = 2
_IDS_OPERATORS = frozenset(chr(cp) for cp in range(0x2FF0, 0x3000))
_ROOT = Path(__file__).resolve().parents[2]


@functools.lru_cache(maxsize=1)
def _ids_decompositions() -> dict[str, str]:
    path = _ROOT / "training_data" / "ids_decomposition.txt"
    result = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.startswith(";"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3 and len(parts[1]) == 1:
            result[parts[1]] = parts[2]
    return result


def _ids_tokens(value: str) -> list[str]:
    tokens = []
    i = 0
    while i < len(value):
        if value[i] == "&":
            end = value.find(";", i + 1)
            if end < 0:
                tokens.append(value[i:])
                break
            tokens.append(value[i:end + 1])
            i = end + 1
        else:
            tokens.append(value[i])
            i += 1
    return tokens


def _component_count(char: str, stack: frozenset[str]) -> int:
    if char in stack:
        return 1
    decomposition = _ids_decompositions().get(char)
    if not decomposition or decomposition == char:
        return 1
    child_stack = stack | {char}
    count = 0
    for token in _ids_tokens(decomposition):
        if token in _IDS_OPERATORS:
            continue
        count += (_component_count(token, child_stack)
                  if len(token) == 1 else 1)
    return min(count or 1, 64)


@functools.lru_cache(maxsize=32_768)
def han_component_count(char: str) -> int:
    """Return a recursive IDS leaf-count proxy for glyph complexity."""
    return _component_count(char, frozenset())


@functools.lru_cache(maxsize=1)
def _direct_han_chars() -> frozenset[str]:
    path = _ROOT / "training_data" / "corpora" / "cjk_vocab.txt"
    if not path.exists():
        return frozenset()
    return frozenset(line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                     if len(line.strip()) == 1)


@functools.lru_cache(maxsize=1)
def _dense_direct_chars() -> frozenset[str]:
    path = _ROOT / "training_data" / "corpora" / "han_dense_chars.txt"
    if not path.exists():
        return frozenset()
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return frozenset("".join(line for line in lines if not line.startswith("#")))


def _complexity_script(char: str) -> str:
    if char in _dense_direct_chars():
        return HAN_DENSE
    if char in _direct_han_chars():
        return HAN_SPARSE
    return (HAN_DENSE if han_component_count(char) > _DENSE_COMPONENT_THRESHOLD
            else HAN_SPARSE)


@functools.lru_cache(maxsize=1)
def _visual_route_overrides() -> dict[str, str]:
    """Route rare ALT characters with their visual leaf prototype."""
    path = _ROOT / "training_data" / "corpora" / "cjk_visual_mapping.tsv"
    result = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            result[parts[0]] = _complexity_script(parts[2])
    return result


@functools.lru_cache(maxsize=32_768)
def han_script_for_char(char: str) -> str:
    """Return the Han subscript used to route and encode ``char``."""
    return _visual_route_overrides().get(char, _complexity_script(char))
