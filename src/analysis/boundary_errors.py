"""How many errors sit at the edges of a segment, where alignment problems show.

    python -m src.analysis.boundary_errors results/evaluations/blind_beam.jsonl

MUCS segments come with timestamps that do not always match their reference
text, so a word can be in the audio but in the neighbouring segment's reference,
or the other way round. Those errors land on the first or last word. This counts
deletions of the first or last reference word and insertions before the first
or after the last one, using the same normalised text as the scorer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.metrics.official import load_maps, normalise_hypothesis, normalise_reference
from src.metrics.wer import DEL, INS, SUB, align


def edge_errors(reference: list[str], hypothesis: list[str]) -> dict[str, int]:
    """Counts of each error type, split into edge and interior."""
    result = {"edge_deletions": 0, "edge_insertions": 0, "deletions": 0, "insertions": 0, "substitutions": 0}
    operations = align(reference, hypothesis)
    last_reference = len(reference) - 1
    # Insertions are at an edge when no reference word has been matched before
    # them (start) or none remains after them (end).
    seen_reference = 0
    for operation, ref_index, _ in operations:
        if operation == DEL:
            result["deletions"] += 1
            result["edge_deletions"] += ref_index in (0, last_reference)
        elif operation == INS:
            result["insertions"] += 1
            result["edge_insertions"] += seen_reference == 0 or seen_reference > last_reference
        elif operation == SUB:
            result["substitutions"] += 1
        if ref_index is not None:
            seen_reference = ref_index + 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    maps = load_maps()
    totals = {"edge_deletions": 0, "edge_insertions": 0, "deletions": 0, "insertions": 0,
              "substitutions": 0, "reference_words": 0}
    with args.predictions.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            reference = normalise_reference(row["reference"]).split()
            if not reference:
                continue
            hypothesis = normalise_hypothesis(row["hypothesis"], maps.punctuation).split()
            for key, value in edge_errors(reference, hypothesis).items():
                totals[key] += value
            totals["reference_words"] += len(reference)
    errors = totals["deletions"] + totals["insertions"] + totals["substitutions"]
    edge = totals["edge_deletions"] + totals["edge_insertions"]
    totals.update({
        "edge_share_of_deletions_percent": round(100 * totals["edge_deletions"] / max(totals["deletions"], 1), 1),
        "edge_share_of_insertions_percent": round(100 * totals["edge_insertions"] / max(totals["insertions"], 1), 1),
        "edge_errors_percent_of_reference_words": round(100 * edge / totals["reference_words"], 2),
        "wer": round(100 * errors / totals["reference_words"], 2),
    })
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
