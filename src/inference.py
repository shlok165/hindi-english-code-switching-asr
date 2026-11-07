"""Single-file wav2vec 2.0 CTC inference with greedy or KenLM beam decoding.

Greedy inference is self-contained. Beam decoding requires the optional
``pyctcdecode`` and ``kenlm`` packages and a compiled KenLM model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.preprocessing import TARGET_SAMPLE_RATE, load_audio
from src.models.ctc_model import greedy_decode, load_model, output_lengths
from src.models.tokenizer import BLANK, BOS, EOS, UNK


def decoder_labels(vocab: dict[str, int]) -> list[str]:
    """Return labels in model-id order, using a space for the word delimiter."""
    indices = sorted(vocab.values())
    if indices != list(range(len(indices))):
        raise ValueError("vocabulary ids must be contiguous and zero-based")
    labels = [""] * len(vocab)
    for token, index in vocab.items():
        if token == BLANK:
            labels[index] = ""
        else:
            labels[index] = " " if token == "|" else token
    if vocab.get(BLANK) != 0:
        raise ValueError("the CTC blank must be vocabulary id 0")
    return labels


def build_beam_decoder(
    vocab: dict[str, int], kenlm_path: str | Path, lm_weight: float, word_insertion: float
) -> Any:
    path = Path(kenlm_path)
    if not path.is_file():
        raise FileNotFoundError(f"KenLM binary model not found: {path}")
    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError as error:
        raise RuntimeError(
            "KenLM beam decoding requires pyctcdecode and kenlm; see docs/stage_1b_handoff.md"
        ) from error
    return build_ctcdecoder(
        decoder_labels(vocab),
        kenlm_model_path=str(path),
        alpha=lm_weight,
        beta=word_insertion,
    )


def decode_logits(
    logits: torch.Tensor,
    vocab: dict[str, int],
    decoder: Any | None = None,
    beam_width: int = 3000,
) -> str:
    if decoder is None:
        return greedy_decode(logits, torch.tensor([logits.shape[1]]), vocab)[0]

    scores = logits[0].detach().float().cpu().numpy().copy()
    for token in (BOS, EOS, UNK):
        scores[:, vocab[token]] = np.finfo(scores.dtype).min
    return " ".join(decoder.decode(scores, beam_width=beam_width).split())


@torch.inference_mode()
def transcribe_audio(
    audio_path: str | Path,
    model: torch.nn.Module,
    vocab: dict[str, int],
    feature_extractor,
    device: torch.device,
    start: float = 0.0,
    end: float | None = None,
    decoder: Any | None = None,
    beam_width: int = 3000,
) -> str:
    """Transcribe one audio range; single-item inference avoids group-norm drift."""
    audio = load_audio(str(audio_path), start, end, TARGET_SAMPLE_RATE)
    features = feature_extractor(
        audio,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_attention_mask=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in features.items()}
    model.eval()
    logits = model(**inputs).logits
    if decoder is None:
        length = output_lengths(model, inputs["attention_mask"].sum(-1))
        return greedy_decode(logits, length, vocab)[0]
    return decode_logits(logits, vocab, decoder, beam_width)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--start", type=float, default=0.0, help="segment start in seconds")
    parser.add_argument("--end", type=float, help="segment end in seconds")
    parser.add_argument("--kenlm", type=Path, help="compiled KenLM model; enables beam decoding")
    parser.add_argument("--beam-width", type=int, default=3000)
    parser.add_argument("--lm-weight", type=float, default=0.5)
    parser.add_argument("--word-insertion", type=float, default=1.0)
    args = parser.parse_args()
    if args.beam_width < 1:
        parser.error("--beam-width must be positive")
    if args.start < 0 or (args.end is not None and args.end <= args.start):
        parser.error("audio segment must satisfy 0 <= start < end")

    model, vocab, extractor = load_model(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    decoder = (
        build_beam_decoder(vocab, args.kenlm, args.lm_weight, args.word_insertion)
        if args.kenlm
        else None
    )
    print(
        transcribe_audio(
            args.audio, model, vocab, extractor, device, args.start, args.end, decoder, args.beam_width
        )
    )


if __name__ == "__main__":
    main()
