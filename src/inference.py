"""Single-file wav2vec 2.0 CTC inference with greedy or KenLM beam decoding.

Greedy inference is self-contained. Beam decoding also needs the optional
``pyctcdecode`` and ``kenlm`` packages and a compiled KenLM model.

    python -m src.inference --checkpoint data/checkpoints/baseline_rtx3050/best --audio a.wav
    python -m src.inference --checkpoint ... --audio a.wav --kenlm data/language_model/train_5gram.klm

Audio is processed one utterance at a time. This checkpoint uses group
normalisation, so padding in a batch would change the features slightly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from src.beam_search import BeamSettings, beam_decode, build_beam_decoder, decoder_labels  # noqa: F401
from src.data.preprocessing import TARGET_SAMPLE_RATE, load_audio
from src.models.ctc_model import greedy_decode, load_model
from src.utils.config import data_path, load_config


@torch.inference_mode()
def log_probabilities(audio, model: torch.nn.Module, feature_extractor, device: torch.device) -> torch.Tensor:
    """Return (frames, vocab) log-probabilities for one float32 waveform."""
    values = feature_extractor(
        audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
    ).input_values.to(device)
    model.eval()
    return torch.log_softmax(model(values).logits[0].float(), dim=-1)


def transcribe_audio(
    audio_path: str | Path,
    model: torch.nn.Module,
    vocab: dict[str, int],
    feature_extractor,
    device: torch.device,
    start: float = 0.0,
    end: float | None = None,
    decoder: Any | None = None,
    settings: BeamSettings | None = None,
) -> str:
    """Transcribe one audio range, greedily unless a beam decoder is given."""
    audio = load_audio(str(audio_path), start, end, TARGET_SAMPLE_RATE)
    scores = log_probabilities(audio, model, feature_extractor, device)
    if decoder is None:
        return greedy_decode(scores.unsqueeze(0), torch.tensor([len(scores)]), vocab)[0]
    return beam_decode(scores.cpu().numpy(), vocab, decoder, settings or BeamSettings())


def settings_from_config(path: str | Path) -> tuple[BeamSettings, str]:
    decoder = load_config(path)["decoder"]
    fields = BeamSettings.__dataclass_fields__
    return BeamSettings(**{k: v for k, v in decoder.items() if k in fields}), decoder["kenlm_model"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--start", type=float, default=0.0, help="segment start in seconds")
    parser.add_argument("--end", type=float, help="segment end in seconds")
    parser.add_argument("--kenlm", type=Path, help="compiled KenLM model; enables beam decoding")
    parser.add_argument("--decode-config", default="configs/decode.yaml")
    args = parser.parse_args()
    if args.start < 0 or (args.end is not None and args.end <= args.start):
        parser.error("audio segment must satisfy 0 <= start < end")

    model, vocab, extractor = load_model(data_path(args.checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    settings, _ = settings_from_config(args.decode_config)
    decoder = build_beam_decoder(vocab, args.kenlm, settings) if args.kenlm else None
    print(transcribe_audio(args.audio, model, vocab, extractor, device, args.start, args.end, decoder, settings))


if __name__ == "__main__":
    main()
