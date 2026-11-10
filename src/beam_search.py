"""KenLM-assisted CTC beam search over cached log-probabilities.

Scoring is done by pyctcdecode. For each candidate word it adds the acoustic
log-probability, `alpha` times the KenLM log-probability, `beta` for each word
(the word insertion score), and `unk_score_offset` for out-of-vocabulary words.

This module imports neither torch nor transformers, which keeps each decoding
worker small. Decoding in pure Python is slow, so utterances run in parallel
processes that each hold their own decoder.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

BLANK, DELIMITER = "<pad>", "|"
HIDDEN_TOKENS = ("<s>", "</s>", "<unk>")
MASKED_LOG_PROB = -34.0  # Below pyctcdecode's clip, so these tokens cannot win


@dataclass(frozen=True)
class BeamSettings:
    alpha: float = 0.5
    beta: float = 1.0
    unk_score_offset: float = -10.0
    beam_width: int = 100
    beam_prune_logp: float = -10.0
    token_min_logp: float = -5.0


def decoder_labels(vocab: dict[str, int]) -> list[str]:
    """Return labels in model-id order, using a space for the word delimiter."""
    indices = sorted(vocab.values())
    if indices != list(range(len(indices))):
        raise ValueError("vocabulary ids must be contiguous and zero-based")
    if vocab.get(BLANK) != 0:
        raise ValueError("the CTC blank must be vocabulary id 0")
    labels = [""] * len(vocab)
    for token, index in vocab.items():
        labels[index] = "" if token == BLANK else " " if token == DELIMITER else token
    return labels


def read_unigrams(kenlm_path: str | Path) -> list[str]:
    """Words known to the LM. A binary LM cannot list them, so the decoder needs this."""
    path = Path(kenlm_path).with_name("unigrams.txt")
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run python -m src.language_model")
    return path.read_text(encoding="utf-8").split()


LM_BACKENDS = ("auto", "kenlm", "arpa")


def kenlm_available() -> bool:
    try:
        import kenlm  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_lm_backend(backend: str) -> str:
    """`auto` means real KenLM when it is installed, else the pure-Python ARPA reader."""
    if backend not in LM_BACKENDS:
        raise ValueError(f"unknown LM backend {backend!r}; choose one of {LM_BACKENDS}")
    if backend == "auto":
        return "kenlm" if kenlm_available() else "arpa"
    if backend == "kenlm" and not kenlm_available():
        raise RuntimeError(
            "the kenlm package is not installed. Use --lm-backend arpa (pure Python, same model "
            "from its .arpa file) or install kenlm; see docs/cpu_setup.md"
        )
    return backend


def arpa_path_for(kenlm_path: str | Path) -> Path:
    """The ARPA file a binary .klm was built from: same name, .arpa suffix."""
    path = Path(kenlm_path)
    return path if path.suffix == ".arpa" else path.with_suffix(".arpa")


def build_beam_decoder(
    vocab: dict[str, int], kenlm_path: str | Path, settings: BeamSettings, backend: str = "auto"
) -> Any:
    """A pyctcdecode decoder over the n-gram model at kenlm_path.

    With the `kenlm` backend, kenlm_path is the .klm (or .arpa) that KenLM loads.
    With `arpa`, the .arpa beside it is read by src.arpa_lm instead.
    """
    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError as error:
        raise RuntimeError(
            "KenLM beam decoding requires pyctcdecode: pip install --no-deps pyctcdecode, "
            "then pip install pygtrie hypothesis; see docs/cpu_setup.md"
        ) from error
    backend = resolve_lm_backend(backend)
    path = Path(kenlm_path)
    if backend == "arpa":
        from src.arpa_lm import build_arpa_decoder

        arpa = arpa_path_for(path)
        if not arpa.is_file():
            raise FileNotFoundError(f"ARPA language model not found: {arpa} (the ARPA backend reads it "
                                    f"instead of {path.name})")
        return build_arpa_decoder(decoder_labels(vocab), arpa, read_unigrams(path), settings.alpha,
                                  settings.beta, settings.unk_score_offset)
    if not path.is_file():
        raise FileNotFoundError(f"KenLM model not found: {path}")
    return build_ctcdecoder(
        decoder_labels(vocab),
        kenlm_model_path=str(path),
        unigrams=read_unigrams(path),
        alpha=settings.alpha,
        beta=settings.beta,
        unk_score_offset=settings.unk_score_offset,
    )


def beam_decode(log_probs: np.ndarray, vocab: dict[str, int], decoder: Any, settings: BeamSettings) -> str:
    """Decode one (frames, vocab) array of log-probabilities or logits."""
    scores = np.array(log_probs, dtype=np.float32)
    for token in HIDDEN_TOKENS:
        scores[:, vocab[token]] = MASKED_LOG_PROB
    text = decoder.decode(
        scores,
        beam_width=settings.beam_width,
        beam_prune_logp=settings.beam_prune_logp,
        token_min_logp=settings.token_min_logp,
    )
    return " ".join(text.split())


def greedy_from_log_probs(log_probs: np.ndarray, vocab: dict[str, int]) -> str:
    """Best path: merge repeats, then drop blanks and tokens that are never targets."""
    inverse = {index: token for token, index in vocab.items()}
    best = np.asarray(log_probs).argmax(axis=1)
    kept = best[np.insert(best[1:] != best[:-1], 0, True)]
    hidden = {BLANK, *HIDDEN_TOKENS}
    tokens = (inverse[int(i)] for i in kept)
    text = "".join(t for t in tokens if t not in hidden).replace(DELIMITER, " ")
    return " ".join(text.split())


class LogProbCache:
    """Per-utterance log-probabilities stored in one memory-mapped array."""

    def __init__(self, directory: str | Path, split: str) -> None:
        directory = Path(directory)
        self.index = json.loads((directory / f"{split}.json").read_text(encoding="utf-8"))
        self.array = np.load(directory / f"{split}.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, position: int) -> np.ndarray:
        entry = self.index[position]
        return self.array[entry["offset"] : entry["offset"] + entry["frames"]]


_worker: dict[str, Any] = {}


def _start_worker(directory: str, split: str, vocab: dict[str, int], kenlm_path: str) -> None:
    _worker["cache"] = LogProbCache(directory, split)
    _worker["vocab"] = vocab
    _worker["decoder"] = build_beam_decoder(vocab, kenlm_path, BeamSettings())


def _decode_position(job: tuple[int, dict]) -> str:
    position, values = job
    settings = BeamSettings(**values)
    _worker["decoder"].reset_params(
        alpha=settings.alpha, beta=settings.beta, unk_score_offset=settings.unk_score_offset
    )
    return beam_decode(_worker["cache"][position], _worker["vocab"], _worker["decoder"], settings)


class BeamPool:
    """Worker processes that decode cached utterances with any settings."""

    def __init__(self, directory: str | Path, split: str, vocab: dict[str, int],
                 kenlm_path: str | Path, workers: int) -> None:
        self.pool = ProcessPoolExecutor(
            workers, initializer=_start_worker, initargs=(str(directory), split, vocab, str(kenlm_path))
        )

    def decode(self, positions: Iterable[int], settings: BeamSettings) -> list[str]:
        values = asdict(settings)
        jobs = [(position, values) for position in positions]
        return list(self.pool.map(_decode_position, jobs, chunksize=4))

    def __enter__(self) -> "BeamPool":
        return self

    def __exit__(self, *_: object) -> None:
        self.pool.shutdown()
