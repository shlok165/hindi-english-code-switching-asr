"""Run single-utterance inference and write complete split artifacts.

Local WER/CER use cleaned manifest text and are for diagnostics. Test and blind
headline scores must be computed with the official scorer and its raw references.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src import ROOT
from src.data.dataset import read_manifest
from src.inference import build_beam_decoder, transcribe_audio
from src.metrics.cer import score as score_cer
from src.metrics.wer import score as score_wer
from src.models.ctc_model import load_model
from src.utils.config import data_path


def evaluate(
    split: str,
    checkpoint: Path,
    manifest: Path,
    output_dir: Path,
    kenlm_path: Path | None = None,
    beam_width: int = 3000,
    lm_weight: float = 0.5,
    word_insertion: float = 1.0,
) -> dict[str, Any]:
    if split not in {"val", "test", "blind"}:
        raise ValueError("evaluation split must be val, test, or blind")
    entries = read_manifest(manifest)
    ids = [entry["id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{manifest} contains duplicate utterance ids")

    model, vocab, extractor = load_model(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    decoder = (
        build_beam_decoder(vocab, kenlm_path, lm_weight, word_insertion)
        if kenlm_path is not None
        else None
    )
    decoder_name = "kenlm_beam" if decoder is not None else "greedy"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{split}_{decoder_name}.jsonl"
    hypotheses_path = output_dir / f"{split}_{decoder_name}.hyp"
    pairs = []

    with predictions_path.open("w", encoding="utf-8", newline="\n") as predictions, \
            hypotheses_path.open("w", encoding="utf-8", newline="\n") as hypotheses:
        for entry in entries:
            audio_path = data_path(entry["audio"])
            if not audio_path.is_file():
                raise FileNotFoundError(f"{entry['id']}: audio not found: {audio_path}")
            prediction = transcribe_audio(
                audio_path,
                model,
                vocab,
                extractor,
                device,
                entry["start"],
                entry["end"],
                decoder,
                beam_width,
            )
            reference = entry["text"]
            pairs.append((reference, prediction))
            predictions.write(
                json.dumps(
                    {
                        "id": entry["id"],
                        "split": split,
                        "duration": entry["duration"],
                        "reference": reference,
                        "reference_raw": entry.get("text_raw", ""),
                        "hypothesis": prediction,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            hypotheses.write(f"{entry['id']} {prediction}\n")

    wer = score_wer(pairs)
    cer = score_cer(pairs)
    summary = {
        "split": split,
        "alignment": "UnA",
        "decoder": decoder_name,
        "utterances": len(entries),
        "wer_cleaned_percent": wer.wer,
        "cer_cleaned_percent": cer.wer,
        "reference_words": wer.counts.ref_words,
        "reference_characters": cer.counts.ref_words,
        "hypotheses": str(hypotheses_path),
        "predictions": str(predictions_path),
        "official_wer_twer": "not computed; use the official scorer on raw references",
    }
    summary_path = output_dir / f"{split}_{decoder_name}_metrics.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("val", "test", "blind"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "evaluations")
    parser.add_argument("--kenlm", type=Path)
    parser.add_argument("--beam-width", type=int, default=3000)
    parser.add_argument("--lm-weight", type=float, default=0.5)
    parser.add_argument("--word-insertion", type=float, default=1.0)
    args = parser.parse_args()
    if args.beam_width < 1:
        parser.error("--beam-width must be positive")
    if args.manifest is None:
        args.manifest = data_path("data/processed/manifests") / f"{args.split}.jsonl"
    evaluate(
        args.split,
        args.checkpoint,
        args.manifest,
        args.output_dir,
        args.kenlm,
        args.beam_width,
        args.lm_weight,
        args.word_insertion,
    )


if __name__ == "__main__":
    main()
