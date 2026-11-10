"""wav2vec 2.0 CTC inference with greedy or KenLM beam decoding, on CPU or GPU.

    python -m src.inference --audio a.wav
    python -m src.inference --audio a.wav --decoder beam --json
    python -m src.inference --manifest audio.csv --output predictions.jsonl --decoder beam

Defaults, including the final model (iteration 4) and its decoder settings, come
from configs/inference.yaml. The older form still works:

    python -m src.inference --checkpoint data/checkpoints/iteration4/best --audio a.wav \
        --kenlm data/language_model/train_5gram.klm --decode-config configs/decode_iteration4.yaml

From Python, load once and transcribe many files:

    from src.inference import Transcriber
    transcriber = Transcriber.from_config(decoder="beam")
    print(transcriber.transcribe("a.wav").text)

Greedy decoding needs only torch and transformers. Beam decoding also needs
pyctcdecode, plus either the `kenlm` package or, without it, the pure-Python
ARPA reader in src.arpa_lm (see docs/cpu_setup.md).

Audio is processed one utterance at a time. This checkpoint uses group
normalisation, so padding in a batch would change the features slightly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch

from src.beam_search import (  # noqa: F401  (decoder_labels is re-exported for tests)
    LM_BACKENDS, BeamSettings, beam_decode, build_beam_decoder, decoder_labels, resolve_lm_backend,
)
from src.data.preprocessing import TARGET_SAMPLE_RATE, load_audio
from src.models.ctc_model import greedy_decode, load_model
from src.models.tokenizer import BLANK, BOS, DELIMITER, EOS, UNK
from src.utils.config import data_path, load_config, resolve

DEFAULT_CONFIG = "configs/inference.yaml"
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
CHECKPOINT_FILES = ("config.json", "preprocessor_config.json", "vocab.json")
END_TOLERANCE = 0.01  # seconds; manifest end times are rounded


class InferenceError(Exception):
    """A problem with the inputs or artifacts, reported without a traceback."""


class ArtifactError(InferenceError, FileNotFoundError):
    """A checkpoint, configuration or language-model file is missing or unusable."""


class AudioValidationError(InferenceError, ValueError):
    """The audio cannot be transcribed: missing, corrupt, empty, too short or long, or not finite."""


@dataclass(frozen=True)
class AudioLimits:
    min_seconds: float = 0.1
    max_seconds: float = 60.0
    silence_peak: float = 1e-4


@dataclass
class Transcription:
    audio: str
    text: str
    decoder: str
    start: float
    end: float
    duration: float
    sample_rate: int
    lm_backend: str | None = None
    processing_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Existing single-utterance API (src.check_inference relies on these) ------------

@torch.inference_mode()
def log_probabilities(audio, model: torch.nn.Module, feature_extractor, device: torch.device) -> torch.Tensor:
    """Return (frames, vocab) log-probabilities for one float32 waveform."""
    values = feature_extractor(
        audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
    ).input_values.to(device)
    model.eval()
    return torch.log_softmax(model(values).logits[0].float(), dim=-1)


def decode_scores(scores: torch.Tensor, vocab: dict[str, int], decoder: Any | None,
                  settings: BeamSettings | None) -> str:
    if decoder is None:
        return greedy_decode(scores.unsqueeze(0), torch.tensor([len(scores)]), vocab)[0]
    return beam_decode(scores.cpu().numpy(), vocab, decoder, settings or BeamSettings())


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
    """Transcribe one audio range, greedily unless a beam decoder is given. No validation."""
    audio = load_audio(str(audio_path), start, end, TARGET_SAMPLE_RATE)
    return decode_scores(log_probabilities(audio, model, feature_extractor, device), vocab, decoder, settings)


def settings_from_config(path: str | Path) -> tuple[BeamSettings, str]:
    """Beam settings and the KenLM path named in a decode config such as configs/decode_iteration4.yaml."""
    if not resolve(path).is_file():
        raise ArtifactError(f"decode config not found: {resolve(path)}")
    decoder = load_config(path).get("decoder")
    if not isinstance(decoder, dict) or "kenlm_model" not in decoder:
        raise ArtifactError(f"{path}: expected a 'decoder' section with 'kenlm_model'")
    fields = BeamSettings.__dataclass_fields__
    return BeamSettings(**{k: v for k, v in decoder.items() if k in fields}), decoder["kenlm_model"]


# Validation -------------------------------------------------------------------

def check_checkpoint(directory: str | Path) -> dict[str, int]:
    """Check that a checkpoint directory holds everything load_model needs; return its vocab."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ArtifactError(
            f"checkpoint directory not found: {directory}. The final model belongs at "
            "data/checkpoints/iteration4/best (unpack checkpoints.zip in the repository root)"
        )
    missing = [name for name in CHECKPOINT_FILES if not (directory / name).is_file()]
    if not any((directory / name).is_file() for name in WEIGHT_FILES):
        missing.append(" or ".join(WEIGHT_FILES))
    if missing:
        raise ArtifactError(f"checkpoint {directory} is missing: {', '.join(missing)}")
    try:
        vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"{directory / 'vocab.json'} is not valid JSON: {error}") from error
    absent = [token for token in (BLANK, BOS, EOS, UNK, DELIMITER) if token not in vocab]
    if absent:
        raise ArtifactError(f"{directory / 'vocab.json'} lacks the special tokens {absent}")
    try:
        decoder_labels(vocab)
    except ValueError as error:
        raise ArtifactError(f"{directory / 'vocab.json'}: {error}") from error
    return vocab


def audio_info(path: str | Path) -> Any:
    """soundfile's header info, with missing, empty and unreadable files reported clearly."""
    path = Path(path)
    if not path.is_file():
        raise AudioValidationError(f"audio file not found: {path}")
    if path.stat().st_size == 0:
        raise AudioValidationError(f"audio file is empty (0 bytes): {path}")
    try:
        info = sf.info(str(path))
    except (RuntimeError, sf.LibsndfileError) as error:  # LibsndfileError subclasses RuntimeError
        raise AudioValidationError(f"{path}: corrupted or unsupported audio file ({error})") from error
    if info.frames == 0 or info.samplerate <= 0:
        raise AudioValidationError(f"{path}: the file holds no audio samples")
    return info


def check_range(path: str | Path, info: Any, start: float, end: float | None, limits: AudioLimits) -> float:
    """Validate the [start, end) range against the file and the limits; return the end time."""
    duration = info.frames / info.samplerate
    if start < 0 or (end is not None and end <= start):
        raise AudioValidationError(f"{path}: the audio range must satisfy 0 <= start < end (got {start}-{end})")
    if start >= duration:
        raise AudioValidationError(f"{path}: start {start:.3f} s is past the end of the audio ({duration:.3f} s)")
    if end is not None and end > duration + END_TOLERANCE:
        raise AudioValidationError(f"{path}: end {end:.3f} s is past the end of the audio ({duration:.3f} s)")
    end = duration if end is None else min(end, duration)
    length = end - start
    if length < limits.min_seconds:
        raise AudioValidationError(
            f"{path}: audio is too short to transcribe ({length:.3f} s; minimum {limits.min_seconds} s)")
    if length > limits.max_seconds:
        raise AudioValidationError(
            f"{path}: audio is too long ({length:.1f} s; maximum {limits.max_seconds} s). "
            "Transcribe it in parts with --start and --end")
    return end


def check_samples(path: str | Path, audio: np.ndarray, limits: AudioLimits) -> list[str]:
    """Reject non-finite samples; return warnings, such as for silence or clipping."""
    if not np.isfinite(audio).all():
        bad = int((~np.isfinite(audio)).sum())
        raise AudioValidationError(f"{path}: audio holds {bad} NaN or infinite samples")
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    warnings = []
    if peak < limits.silence_peak:
        warnings.append(f"audio is silent (peak amplitude {peak:.2e} < {limits.silence_peak}); nothing decoded")
    elif peak > 1.0:
        warnings.append(f"samples exceed [-1, 1] (peak {peak:.2f}); the feature extractor normalises them")
    return warnings


def locate(path: str | Path) -> Path:
    """A path as given when it exists from the current directory, else against the repository
    (and MUCS_DATA_ROOT for data/), as configs/inference.yaml paths are."""
    path = Path(path)
    return path if path.exists() else data_path(path)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise InferenceError("--device cuda was requested but CUDA is not available; use --device cpu")
    if name not in ("cpu", "cuda"):
        raise InferenceError(f"unknown device {name!r}; choose auto, cpu or cuda")
    return torch.device(name)


# Reusable transcriber -------------------------------------------------------------

class Transcriber:
    """Load a checkpoint (and optionally a beam decoder) once, then transcribe many files."""

    def __init__(
        self,
        checkpoint: str | Path,
        decoder: str = "greedy",
        decode_config: str | Path | None = None,
        kenlm: str | Path | None = None,
        lm_backend: str = "auto",
        device: str = "auto",
        limits: AudioLimits = AudioLimits(),
    ) -> None:
        if decoder not in ("greedy", "beam"):
            raise InferenceError(f"unknown decoder {decoder!r}; choose greedy or beam")
        self.checkpoint = locate(checkpoint)
        self.decoder_name = decoder
        self.limits = limits
        self.device = resolve_device(device)
        self.vocab = check_checkpoint(self.checkpoint)
        try:
            self.model, _, self.feature_extractor = load_model(self.checkpoint)
        except (OSError, ValueError, RuntimeError) as error:
            raise ArtifactError(f"could not load the checkpoint in {self.checkpoint}: {error}") from error
        if self.model.config.vocab_size != len(self.vocab):
            raise ArtifactError(f"{self.checkpoint}: the model has {self.model.config.vocab_size} outputs "
                                f"but vocab.json has {len(self.vocab)} tokens")
        self.model.to(self.device).eval()

        self.settings: BeamSettings | None = None
        self.beam_decoder: Any | None = None
        self.lm_backend: str | None = None
        self.kenlm_path: Path | None = None
        if decoder == "beam":
            if decode_config is None:
                raise InferenceError("beam decoding needs a decode config, such as configs/decode_iteration4.yaml")
            self.settings, configured = settings_from_config(decode_config)
            self.kenlm_path = locate(kenlm if kenlm is not None else configured)
            try:
                self.lm_backend = resolve_lm_backend(lm_backend)
                self.beam_decoder = build_beam_decoder(self.vocab, self.kenlm_path, self.settings, self.lm_backend)
            except FileNotFoundError as error:
                raise ArtifactError(str(error)) from error
            except (RuntimeError, ValueError) as error:
                raise InferenceError(str(error)) from error

    @classmethod
    def from_config(cls, config: str | Path = DEFAULT_CONFIG, **overrides: Any) -> "Transcriber":
        """Build from configs/inference.yaml; keyword arguments that are not None win."""
        settings = inference_settings(config)
        settings.update({key: value for key, value in overrides.items() if value is not None})
        limits = AudioLimits(float(settings["min_seconds"]), float(settings["max_seconds"]),
                             float(settings["silence_peak"]))
        return cls(settings["checkpoint"], settings["decoder"], settings["decode_config"],
                   settings.get("kenlm"), settings["lm_backend"], settings["device"], limits)

    def transcribe(self, audio_path: str | Path, start: float = 0.0, end: float | None = None) -> Transcription:
        began = time.perf_counter()
        info = audio_info(audio_path)
        end = check_range(audio_path, info, start, end, self.limits)
        audio = load_audio(str(audio_path), start, end, TARGET_SAMPLE_RATE)
        warnings = check_samples(audio_path, audio, self.limits)
        if info.channels > 1:
            warnings.append(f"{info.channels} channels were averaged to mono")
        if info.samplerate != TARGET_SAMPLE_RATE:
            warnings.append(f"resampled from {info.samplerate} Hz to {TARGET_SAMPLE_RATE} Hz")
        if warnings and warnings[0].startswith("audio is silent"):
            text = ""
        else:
            scores = log_probabilities(audio, self.model, self.feature_extractor, self.device)
            text = decode_scores(scores, self.vocab, self.beam_decoder, self.settings)
        return Transcription(
            audio=str(audio_path), text=text, decoder=self.decoder_name, start=start, end=round(end, 6),
            duration=round(end - start, 6), sample_rate=info.samplerate, lm_backend=self.lm_backend,
            processing_seconds=round(time.perf_counter() - began, 3), warnings=warnings,
        )


def inference_settings(config: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    if not resolve(config).is_file():
        raise ArtifactError(f"inference config not found: {resolve(config)}")
    settings = load_config(config).get("inference")
    required = ("checkpoint", "decode_config", "decoder", "lm_backend", "device",
                "min_seconds", "max_seconds", "silence_peak")
    if not isinstance(settings, dict) or any(key not in settings for key in required):
        raise ArtifactError(f"{config}: the 'inference' section must set {', '.join(required)}")
    return dict(settings)


# Manifests -----------------------------------------------------------------------

def read_audio_list(path: Path) -> list[dict[str, Any]]:
    """Rows with `audio` and optional `id`, `start`, `end`, `reference`, from CSV or JSONL."""
    if not path.is_file():
        raise InferenceError(f"manifest not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = list(csv.DictReader(text.splitlines()))
    if not rows or any(not row.get("audio") for row in rows):
        raise InferenceError(f"{path}: every row needs an 'audio' column")
    base = path.parent
    entries = []
    for number, row in enumerate(rows):
        audio = Path(row["audio"])
        entries.append({
            "id": row.get("id") or audio.stem or str(number),
            "audio": audio if audio.is_absolute() or audio.exists() else base / audio,
            "start": float(row.get("start") or 0.0),
            "end": float(row["end"]) if row.get("end") not in (None, "") else None,
            "reference": row.get("reference"),
        })
    return entries


def transcribe_manifest(transcriber: Transcriber, entries: Iterable[dict], output: Path) -> tuple[int, int]:
    """Write one JSON line per entry; a failed entry records its error and the rest continue."""
    done = failed = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            row: dict[str, Any] = {"id": entry["id"], "audio": str(entry["audio"]), "start": entry["start"],
                                   "end": entry["end"]}
            if entry.get("reference") is not None:
                row["reference"] = entry["reference"]
            try:
                result = transcriber.transcribe(entry["audio"], entry["start"], entry["end"])
                row.update(end=result.end, duration=result.duration, hypothesis=result.text,
                           decoder=result.decoder, lm_backend=result.lm_backend, warnings=result.warnings)
                done += 1
            except AudioValidationError as error:
                row.update(hypothesis=None, error=str(error))
                failed += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return done, failed


# Command line ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--audio", nargs="+", type=Path, help="one or more audio files (WAV recommended)")
    inputs.add_argument("--manifest", type=Path, help="CSV or JSONL with audio[,id,start,end,reference]")
    parser.add_argument("--output", type=Path, help="JSONL predictions file; required with --manifest")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="inference defaults (default: %(default)s)")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint directory (default: from --config)")
    parser.add_argument("--start", type=float, default=0.0, help="segment start in seconds (one file only)")
    parser.add_argument("--end", type=float, help="segment end in seconds (one file only)")
    parser.add_argument("--decoder", choices=("greedy", "beam"), help="default: from --config")
    parser.add_argument("--kenlm", type=Path, help="language model to use instead of the decode config's; "
                                                    "implies --decoder beam")
    parser.add_argument("--decode-config", help="beam settings (default: from --config)")
    parser.add_argument("--lm-backend", choices=LM_BACKENDS, help="default: from --config")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="default: from --config")
    parser.add_argument("--json", action="store_true", help="print one JSON object per file")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:  # the weight-loading progress bar would interleave with --json output on a terminal
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
    except ImportError:
        pass
    if args.audio and len(args.audio) > 1 and (args.start or args.end is not None):
        parser.error("--start and --end apply to a single --audio file")
    if args.manifest and not args.output:
        parser.error("--manifest needs --output")
    decoder = "beam" if args.kenlm else args.decoder
    if args.kenlm and args.decoder == "greedy":
        parser.error("--kenlm needs --decoder beam")
    try:
        transcriber = Transcriber.from_config(
            args.config, checkpoint=args.checkpoint, decoder=decoder, decode_config=args.decode_config,
            kenlm=args.kenlm, lm_backend=args.lm_backend, device=args.device,
        )
        if args.manifest:
            done, failed = transcribe_manifest(transcriber, read_audio_list(args.manifest), args.output)
            print(f"transcribed {done}, failed {failed}; wrote {args.output}", file=sys.stderr)
            return 1 if failed else 0
        failed = 0
        for path in args.audio:
            try:
                result = transcriber.transcribe(path, args.start, args.end)
            except AudioValidationError as error:
                failed += 1
                if args.json:
                    print(json.dumps({"audio": str(path), "error": str(error)}, ensure_ascii=False))
                else:
                    print(f"error: {error}", file=sys.stderr)
                continue
            if args.json:
                print(json.dumps(result.to_dict(), ensure_ascii=False))
            else:
                for warning in result.warnings:
                    print(f"warning: {path}: {warning}", file=sys.stderr)
                print(result.text if len(args.audio) == 1 else f"{path}\t{result.text}")
        return 1 if failed else 0
    except InferenceError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
