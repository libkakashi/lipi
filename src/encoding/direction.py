"""Writing-direction helpers shared by training, evaluation, and inference."""

from __future__ import annotations

from typing import TypeVar


RTL_SCRIPTS = frozenset({"arabic", "hebrew"})
# Stored in checkpoints. Version 1 traversed every segment left-to-right.
CTC_DIRECTION_VERSION = 2

T = TypeVar("T")


def is_rtl_script(script: str) -> bool:
    """Return whether CTC should traverse this script right-to-left."""
    return script in RTL_SCRIPTS


def ctc_time_order(sequence: T, script: str) -> T:
    """Put a time-major sequence in the script's logical reading order.

    Images and encoder frames are stored left-to-right. For RTL scripts the
    first logical character is at the right edge, so CTC must consume the
    frame slice in reverse. Torch tensors and NumPy arrays both support the
    operations used here; lists fall back to normal slicing.
    """
    if not is_rtl_script(script):
        return sequence
    flip = getattr(sequence, "flip", None)
    if flip is not None:
        return flip(0)
    return sequence[::-1]
