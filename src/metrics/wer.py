"""Word error rate with an explicit alignment.

WER = (S + D + I) / N, summed over utterances, which is what Kaldi's
compute-wer reports. Alignments are kept because T-WER, CS-WER and the error
analyses all need to know which reference word each error landed on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

MATCH, SUB, DEL, INS = "=", "S", "D", "I"


@dataclass
class Counts:
    ref_words: int = 0
    matches: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        return 100.0 * self.errors / self.ref_words if self.ref_words else 0.0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(
            self.ref_words + other.ref_words,
            self.matches + other.matches,
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
        )


def align(ref: Sequence[str], hyp: Sequence[str]) -> list[tuple[str, int | None, int | None]]:
    """Return (op, ref_index, hyp_index) triples for a minimum-edit alignment.

    Ties prefer a match or substitution over an insertion or deletion, which
    keeps words paired positionally when several alignments cost the same.
    """
    rows, cols = len(ref) + 1, len(hyp) + 1
    cost = [[0] * cols for _ in range(rows)]
    for j in range(cols):
        cost[0][j] = j
    for i in range(1, rows):
        cost[i][0] = i
        previous, current, word = cost[i - 1], cost[i], ref[i - 1]
        for j in range(1, cols):
            diagonal = previous[j - 1] + (word != hyp[j - 1])
            current[j] = min(diagonal, previous[j] + 1, current[j - 1] + 1)

    ops: list[tuple[str, int | None, int | None]] = []
    i, j = rows - 1, cols - 1
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append((MATCH if ref[i - 1] == hyp[j - 1] else SUB, i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            ops.append((DEL, i - 1, None))
            i -= 1
        else:
            ops.append((INS, None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def count(ref: Sequence[str], hyp: Sequence[str]) -> Counts:
    ops = [op for op, _, _ in align(ref, hyp)]
    return Counts(
        ref_words=len(ref),
        matches=ops.count(MATCH),
        substitutions=ops.count(SUB),
        deletions=ops.count(DEL),
        insertions=ops.count(INS),
    )


@dataclass
class Score:
    counts: Counts = field(default_factory=Counts)
    utterances: int = 0

    @property
    def wer(self) -> float:
        return self.counts.wer


def score(pairs: Iterable[tuple[str, str]]) -> Score:
    """Score (reference, hypothesis) strings, whitespace-tokenised."""
    total, utterances = Counts(), 0
    for ref, hyp in pairs:
        total = total + count(ref.split(), hyp.split())
        utterances += 1
    return Score(total, utterances)
