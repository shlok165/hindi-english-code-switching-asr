"""Character error rate on the project's consistently cleaned transcripts.

Sentence punctuation is stripped and whitespace is collapsed using the same
normalizer as training. Spaces count as characters; this makes word-boundary
errors visible in CER as well as WER.
"""

from __future__ import annotations

from typing import Iterable

from src.data.preprocessing import clean_transcript
from src.metrics.wer import Score, count


def score(pairs: Iterable[tuple[str, str]]) -> Score:
    """Score (reference, hypothesis) strings at Unicode-code-point level."""
    total = count([], [])
    utterances = 0
    for reference, hypothesis in pairs:
        reference_chars = list(clean_transcript(reference))
        hypothesis_chars = list(clean_transcript(hypothesis))
        total = total + count(reference_chars, hypothesis_chars)
        utterances += 1
    return Score(total, utterances)
