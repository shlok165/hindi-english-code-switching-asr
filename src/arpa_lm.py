"""Pure-Python ARPA n-gram model for pyctcdecode, used when `kenlm` is missing.

PyPI has no Windows wheel for `kenlm`, and building it needs a C++ toolchain
and the Windows SDK. This module lets beam search use the same trained
language model without native code. It reads the ARPA file the `.klm` was
built from (`train_5gram.arpa` beside `train_5gram.klm`) and implements the
subset of `kenlm.Model` that pyctcdecode calls.

Scoring is standard ARPA back-off, as KenLM does it:

    log10 p(w | h) = log10 P(h w)                     if the n-gram h w is listed
                   = backoff(h) + log10 p(w | h[1:])  otherwise

An out-of-vocabulary word scores as `<unk>` and leaves an empty context, which
is what KenLM's state does after `<unk>`. `kenlm.Model.__contains__` is False
for `<unk>`, and so is ours.

Agreement with real KenLM is measured, not assumed: `python -m src.arpa_lm`
writes per-word scores for a sample of sentences, `scripts/kenlm_reference.py`
writes KenLM's scores for the same sentences from the `.klm`, and
`scripts/compare_lm_scores.py` compares them (see docs/backend.md).
Loading the 5-gram ARPA takes tens of seconds and a few hundred MB of memory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Collection, Iterable

START, END, UNKNOWN = "<s>", "</s>", "<unk>"
Context = tuple[str, ...]


class ArpaFormatError(ValueError):
    """The file is not an ARPA n-gram model this reader understands."""


def read_arpa(path: str | Path) -> tuple[int, dict[str, float], dict[str, float]]:
    """Return (order, log10 probabilities, log10 back-offs), keyed by space-joined n-gram."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ARPA language model not found: {path}")
    declared: dict[int, int] = {}
    probabilities: dict[str, float] = {}
    backoffs: dict[str, float] = {}
    order = 0
    section = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line == "\\data\\":
                section = "data"
                continue
            if line == "\\end\\":
                break
            if line.startswith("\\") and line.endswith("-grams:"):
                order = int(line[1:].split("-")[0])
                section = "grams"
                continue
            if section == "data" and line.startswith("ngram "):
                level, count = line[6:].split("=")
                declared[int(level)] = int(count)
                continue
            if section != "grams":
                continue
            fields = line.split("\t")
            if len(fields) not in (2, 3):
                raise ArpaFormatError(f"{path}: malformed n-gram line: {line[:80]!r}")
            probabilities[fields[1]] = float(fields[0])
            if len(fields) == 3 and float(fields[2]) != 0.0:
                backoffs[fields[1]] = float(fields[2])
    if not declared or order != max(declared):
        raise ArpaFormatError(f"{path}: missing \\data\\ header or n-gram sections")
    found: dict[int, int] = {}
    for key in probabilities:
        level = key.count(" ") + 1
        found[level] = found.get(level, 0) + 1
    if found != declared:
        raise ArpaFormatError(f"{path}: header declares {declared} n-grams but the file holds {found}")
    if UNKNOWN not in probabilities:
        raise ArpaFormatError(f"{path}: no <unk> unigram")
    return order, probabilities, backoffs


class ArpaModel:
    """The part of `kenlm.Model` that pyctcdecode uses, over an ARPA file.

    States are tuples of context words, most recent last, at most order - 1 long.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path).encode("utf-8")  # kenlm.Model.path is bytes
        self.order, self._log10, self._backoff = read_arpa(path)

    def __contains__(self, word: str) -> bool:
        return word != UNKNOWN and word in self._log10

    def begin_sentence(self) -> Context:
        return (START,)

    def null_context(self) -> Context:
        return ()

    def score_word(self, context: Context, word: str) -> tuple[float, Context]:
        """log10 p(word | context) and the context to score the next word with."""
        known = word in self
        token = word if known else UNKNOWN
        history = context[len(context) - min(len(context), self.order - 1):]
        total = 0.0
        for start in range(len(history) + 1):
            suffix = history[start:]
            probability = self._log10.get(" ".join(suffix + (token,)))
            if probability is not None:
                total += probability
                break
            total += self._backoff.get(" ".join(suffix), 0.0)
        next_context = (history + (token,))[-(self.order - 1):] if known else ()
        return total, next_context

    def word_scores(self, sentence: str, bos: bool = True, eos: bool = True) -> list[float]:
        """Per-word log10 probabilities, like the first field of `kenlm.Model.full_scores`."""
        context = self.begin_sentence() if bos else self.null_context()
        scores = []
        for word in sentence.split() + ([END] if eos else []):
            score, context = self.score_word(context, word)
            scores.append(score)
        return scores

    def score_sentence(self, sentence: str, bos: bool = True, eos: bool = True) -> float:
        """Total log10 probability, like `kenlm.Model.score(sentence, bos, eos)`."""
        return sum(self.word_scores(sentence, bos, eos))


def build_arpa_decoder(
    labels: list[str],
    arpa_path: str | Path,
    unigrams: Collection[str],
    alpha: float,
    beta: float,
    unk_score_offset: float,
    model: ArpaModel | None = None,
):
    """A pyctcdecode BeamSearchDecoderCTC whose LM is an ArpaModel."""
    from pyctcdecode.alphabet import Alphabet, verify_alphabet_coverage
    from pyctcdecode.constants import LOG_BASE_CHANGE_FACTOR
    from pyctcdecode.decoder import BeamSearchDecoderCTC
    from pyctcdecode.language_model import LanguageModel

    class ArpaLanguageModel(LanguageModel):
        """pyctcdecode's LanguageModel with kenlm.State replaced by context tuples."""

        def get_start_state(self) -> Context:
            return self._kenlm_model.begin_sentence() if self.score_boundary else self._kenlm_model.null_context()

        def _get_raw_end_score(self, start_state: Context) -> float:
            return self._kenlm_model.score_word(start_state, END)[0] if self.score_boundary else 0.0

        def score(self, prev_state: Context, word: str, is_last_word: bool = False) -> tuple[float, Context]:
            # Same arithmetic as LanguageModel.score, with ArpaModel in place of BaseScore.
            lm_score, end_state = self._kenlm_model.score_word(prev_state, word)
            if len(self._unigram_set) > 0 and word not in self._unigram_set or word not in self._kenlm_model:
                lm_score += self.unk_score_offset
            if is_last_word:
                lm_score = lm_score + self._get_raw_end_score(end_state)
            lm_score = self.alpha * lm_score * LOG_BASE_CHANGE_FACTOR + self.beta
            return lm_score, end_state

    alphabet =Alphabet.build_alphabet(labels)
    verify_alphabet_coverage(alphabet, unigrams)
    language_model = ArpaLanguageModel(
        model or ArpaModel(arpa_path), unigrams, alpha=alpha, beta=beta, unk_score_offset=unk_score_offset
    )
    return BeamSearchDecoderCTC(alphabet, language_model)


def sample_sentences(text_file: Path, count: int, seed: int) -> list[str]:
    """A reproducible sample of lines, the same lines with their words shuffled
    (which forces back-off to shorter n-grams), and sentences with unseen words."""
    import random

    generator = random.Random(seed)
    lines = [line.strip() for line in text_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    sample = generator.sample(lines, min(count, len(lines)))
    shuffled = []
    for line in sample:
        words = line.split()
        generator.shuffle(words)
        shuffled.append(" ".join(words))
    return sample + shuffled + ["zzqx यह शब्द नहीं है", "click करें qqqqword फिर type", "है है है है है है"]


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score sentences with the pure-Python ARPA model.")
    parser.add_argument("--arpa", type=Path, default=Path("data/language_model/train_5gram.arpa"))
    parser.add_argument("--sentences", type=Path, help="one sentence per line; default: a sample of train.txt")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="JSON lines of {sentence, log10}")
    args = parser.parse_args(list(argv) if argv is not None else None)
    from src.utils.config import data_path

    arpa = data_path(args.arpa)
    if args.sentences:
        sentences = [s.strip() for s in args.sentences.read_text(encoding="utf-8").splitlines() if s.strip()]
    else:
        sentences = sample_sentences(arpa.with_name("train.txt"), args.count, args.seed)
    model = ArpaModel(arpa)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for sentence in sentences:
            words = model.word_scores(sentence)
            row = {"sentence": sentence, "log10": sum(words), "words": words}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"scored {len(sentences)} sentences with {arpa} (order {model.order})", file=sys.stderr)


if __name__ == "__main__":
    main()
