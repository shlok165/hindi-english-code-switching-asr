"""WER and T-WER as defined by the organisers' compute_wer.py.

The scorer ships in the blind archive (`inference/compute_wer.py`), and
evaluation re-runs that script to confirm these numbers. Its rules are:
- lower-case both sides
- apply punct.mapping.txt to the hypothesis only ("slash" becomes "/")
- for T-WER, also apply hin2engwords.txt to the hypothesis only
- skip utterances with an empty reference
- the first entry wins when a mapping key is repeated

Official T-WER is defined only for blind, whose references were annotated so
every English word is in Latin script. Test references were not, so
`approximate_twer` maps both sides instead. That figure is an approximation:
the map was built from blind's vocabulary and misses words only test contains.

The word map comes from the blind references, so it must only ever be used for
scoring. Never use it for training, the LM or tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from src.metrics.wer import score
from src.utils.config import data_path

INFERENCE_DIR = "data/interim/extracted/blind/subtask2_blindtest_wReadme/inference"


def read_mapping(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if parts and parts[0] not in mapping:
            mapping[parts[0]] = " ".join(parts[1:])
    return mapping


@dataclass
class ScoringMaps:
    punctuation: dict[str, str]
    words: dict[str, str]


def load_maps(directory: str | Path = INFERENCE_DIR) -> ScoringMaps:
    root = data_path(directory)
    return ScoringMaps(read_mapping(root / "punct.mapping.txt"), read_mapping(root / "hin2engwords.txt"))


def apply_map(text: str, mapping: Mapping[str, str]) -> str:
    return " ".join(mapping.get(word, word) for word in text.split())


def normalise_reference(text: str) -> str:
    return " ".join(text.lower().split())


def normalise_hypothesis(text: str, punctuation: Mapping[str, str]) -> str:
    return apply_map(" ".join(text.lower().split()), punctuation)


def official_wer(pairs: Iterable[tuple[str, str]], maps: ScoringMaps) -> dict[str, float]:
    """Return WER and T-WER for (raw reference, hypothesis) pairs."""
    plain, mapped = [], []
    for reference, hypothesis in pairs:
        reference = normalise_reference(reference)
        if not reference:
            continue
        hypothesis = normalise_hypothesis(hypothesis, maps.punctuation)
        plain.append((reference, hypothesis))
        mapped.append((reference, apply_map(hypothesis, maps.words)))
    result = score(plain)
    return {"wer": result.wer, "twer": score(mapped).wer, "reference_words": result.counts.ref_words,
            "scored_utterances": result.utterances}


def approximate_twer(pairs: Iterable[tuple[str, str]], maps: ScoringMaps) -> dict[str, float]:
    """T-WER for references that were not annotated: both sides are mapped."""
    mapped = []
    for reference, hypothesis in pairs:
        reference = normalise_reference(reference)
        if not reference:
            continue
        hypothesis = normalise_hypothesis(hypothesis, maps.punctuation)
        mapped.append((apply_map(reference, maps.words), apply_map(hypothesis, maps.words)))
    return {"twer_approximate": score(mapped).wer}
