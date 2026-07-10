"""Writing-direction helpers shared by training, evaluation, and inference."""

from __future__ import annotations

import unicodedata
from typing import TypeVar


RTL_SCRIPTS = frozenset({"arabic", "hebrew"})
# Stored in checkpoints. Version 1 traversed every segment left-to-right.
CTC_DIRECTION_VERSION = 2

_STRONG_RTL_BIDI = frozenset({"R", "AL"})
_NUMBER_BIDI = frozenset({"EN", "AN"})

T = TypeVar("T")


def is_rtl_script(script: str) -> bool:
    """Return whether CTC should traverse this script right-to-left."""
    return script in RTL_SCRIPTS


def is_number_char(ch: str) -> bool:
    """Digits render left-to-right even inside RTL text (bidi EN/AN)."""
    return unicodedata.bidirectional(ch) in _NUMBER_BIDI


def segment_is_rtl(script: str, text: str) -> bool:
    """Return whether CTC should traverse this segment right-to-left.

    A segment of an RTL script is only reversed when its text contains a
    strong RTL character: digit runs (Arabic-Indic/Persian numerals) render
    left-to-right even inside RTL words, so data generation splits them into
    their own segments, which must be consumed in frame order.
    """
    return (is_rtl_script(script)
            and any(unicodedata.bidirectional(ch) in _STRONG_RTL_BIDI
                    for ch in text))


def ctc_time_order(sequence: T, script: str) -> T:
    """Put a time-major sequence in the script's logical reading order.

    Images and encoder frames are stored left-to-right. For RTL scripts the
    first logical character is at the right edge, so CTC must consume the
    frame slice in reverse. Torch tensors and NumPy arrays both support the
    operations used here; lists fall back to normal slicing.

    Used at inference, where the text is unknown before decoding — pair
    with undo_visual_digit_order() on the decoded text. When the segment's
    ground-truth text is known, use ctc_segment_order() instead.
    """
    if not is_rtl_script(script):
        return sequence
    flip = getattr(sequence, "flip", None)
    if flip is not None:
        return flip(0)
    return sequence[::-1]


def ctc_segment_order(sequence: T, script: str, text: str) -> T:
    """ctc_time_order for a segment whose ground-truth text is known."""
    if not segment_is_rtl(script, text):
        return sequence
    flip = getattr(sequence, "flip", None)
    if flip is not None:
        return flip(0)
    return sequence[::-1]


def undo_visual_digit_order(text: str) -> str:
    """Re-reverse digit runs in text decoded from reversed RTL frames.

    Digit runs render left-to-right inside RTL text, so reading the frames
    right-to-left yields each run's digits reversed ("۱۳۹۸" → "۸۹۳۱").
    Reversing every maximal digit run restores logical order.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        if is_number_char(text[i]):
            j = i + 1
            while j < len(text) and is_number_char(text[j]):
                j += 1
            out.append(text[i:j][::-1])
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)
