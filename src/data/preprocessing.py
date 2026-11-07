"""Reusable audio loading and transcript cleaning.

Audio: wav2vec 2.0 expects 16 kHz mono. Amplitude normalisation is left to the
feature extractor (zero mean, unit variance per utterance), which matches how
the pretrained weights were trained. That makes the loudness normalisation in
Chadha et al. redundant, so it is deliberately not applied.

Text: this is the baseline, so scripts are preserved. A word stays in whichever
script the transcriber used. Transliteration and script normalisation are
Stage 2 experiments and must not be added here.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16_000

# Invisible format characters that make identical words compare as different.
INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))
SPACES = re.compile(r"[  -   　]")

# Sentence punctuation has no spoken form, so it is removed. Symbols such as
# * + / @ _ = % stand for spoken words in these programming tutorials ("asterisk",
# "plus"), so they stay, subject to the vocabulary frequency cut-off.
PUNCTUATION = re.compile("[,.;:!?\"'`‘’“”–—…()\\[\\]।॥→]")

# Typists often enter the Devanagari visarga as an ASCII colon. Stripping the
# colon from these words would change them, so it is mapped to the visarga.
VISARGA_STEMS = ("अत", "स्वत", "पुन", "अंतत", "संभवत", "सम्भवत", "विशेषत", "पूर्णत", "साधारणत", "सामान्यत", "मूलत", "क्रमश")
VISARGA = {f"{stem}:": f"{stem}ः" for stem in VISARGA_STEMS}


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Average channels. Accepts (samples,) or (samples, channels)."""
    return audio if audio.ndim == 1 else audio.mean(axis=1)


def resample(audio: np.ndarray, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(source_rate, target_rate)
    return resample_poly(audio, target_rate // divisor, source_rate // divisor).astype(np.float32)


def load_audio(
    path: str, start: float = 0.0, end: float | None = None, target_rate: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """Read [start, end) seconds as float32 mono in [-1, 1] at target_rate."""
    with sf.SoundFile(path) as handle:
        rate = handle.samplerate
        first = round(start * rate)
        last = handle.frames if end is None else min(round(end * rate), handle.frames)
        if not 0 <= first < last:
            raise ValueError(f"{path}: empty or invalid range {start}-{end}")
        handle.seek(first)
        audio = handle.read(last - first, dtype="float32", always_2d=True)
    return resample(to_mono(audio), rate, target_rate)


def valid_duration(seconds: float, minimum: float, maximum: float) -> bool:
    return minimum <= seconds <= maximum


def normalise_unicode(text: str) -> str:
    """NFC, then remove invisible characters and unify spaces.

    NFC splits precomposed nukta letters (U+0958-095F) into consonant + nukta,
    so both spellings of a word such as ज़रूर end up identical.
    """
    return SPACES.sub(" ", unicodedata.normalize("NFC", text)).translate(INVISIBLE)


def clean_transcript(text: str) -> str:
    """Lower-case, restore visargas, remove punctuation, collapse whitespace."""
    tokens = normalise_unicode(text).lower().split()
    restored = " ".join(VISARGA.get(token, token) for token in tokens)
    return " ".join(PUNCTUATION.sub(" ", restored).split())
