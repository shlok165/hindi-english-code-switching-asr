"""Character error rate over Unicode code points.

Spaces count as characters, which makes word-boundary errors visible in CER as
well as WER. Devanagari vowel signs are separate code points, so one wrong
matra costs one character.

`score` cleans both sides with the training normaliser. Pass `clean=False` when
the text has already been normalised for scoring.
"""

from __future__ import annotations

from typing import Iterable

from rapidfuzz.distance import Levenshtein

from src.data.preprocessing import clean_transcript
from src.metrics.wer import Counts, Score


def count_characters(reference: str, hypothesis: str) -> Counts:
    """Edit counts for one pair. rapidfuzz is used because utterances are long."""
    operations = Levenshtein.editops(reference, hypothesis)
    tags = [operation.tag for operation in operations]
    substitutions = tags.count("replace")
    deletions = tags.count("delete")
    return Counts(
        ref_words=len(reference),
        matches=len(reference) - substitutions - deletions,
        substitutions=substitutions,
        deletions=deletions,
        insertions=tags.count("insert"),
    )


def score(pairs: Iterable[tuple[str, str]], clean: bool = True) -> Score:
    """Score (reference, hypothesis) strings at code-point level."""
    total, utterances = Counts(), 0
    for reference, hypothesis in pairs:
        if clean:
            reference, hypothesis = clean_transcript(reference), clean_transcript(hypothesis)
        total = total + count_characters(reference, hypothesis)
        utterances += 1
    return Score(total, utterances)
