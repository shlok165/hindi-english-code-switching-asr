"""Run the acoustic model once per split and save its log-probabilities.

Beam search and LM tuning then never have to touch the GPU again. Every
utterance in the manifest is kept, including those with empty references,
because the official scorer needs a hypothesis for each one.

    python -m src.logit_cache --split val --checkpoint data/checkpoints/baseline_rtx3050/best
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.preprocessing import load_audio
from src.inference import log_probabilities
from src.models.ctc_model import load_model
from src.utils.config import data_path, read_manifest

LOWEST_LOG_PROB = -30.0  # float16 keeps about 0.02 resolution down here


def cache_directory(checkpoint: Path) -> Path:
    return data_path("data/processed/logits") / checkpoint.parent.name / checkpoint.name


def build_cache(split: str, checkpoint: Path, output: Path) -> dict:
    entries = read_manifest(data_path("data/processed/manifests") / f"{split}.jsonl")
    model, vocab, extractor = load_model(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    output.mkdir(parents=True, exist_ok=True)
    chunks, index, offset, began = [], [], 0, time.time()
    for done, entry in enumerate(entries, 1):
        audio = load_audio(str(data_path(entry["audio"])), entry["start"], entry["end"])
        scores = log_probabilities(audio, model, extractor, device).clamp(min=LOWEST_LOG_PROB)
        chunks.append(scores.cpu().numpy().astype(np.float16))
        index.append({"id": entry["id"], "offset": offset, "frames": len(scores)})
        offset += len(scores)
        if done % 500 == 0:
            print(f"{split}: {done}/{len(entries)}", flush=True)
    np.save(output / f"{split}.npy", np.concatenate(chunks))
    (output / f"{split}.json").write_text(json.dumps(index), encoding="utf-8")
    seconds = sum(entry["duration"] for entry in entries)
    elapsed = time.time() - began
    summary = {"split": split, "utterances": len(entries), "frames": offset,
               "audio_seconds": round(seconds, 1), "seconds": round(elapsed, 1)}
    print(summary, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("val", "test", "blind"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--skip-existing", action="store_true",
                        help="leave the split alone if its cache was already written in full")
    args = parser.parse_args()
    checkpoint = data_path(args.checkpoint)
    output = cache_directory(checkpoint)
    # The index is written after the array, so both files mean the cache is complete.
    if args.skip_existing and (output / f"{args.split}.npy").exists() and (output / f"{args.split}.json").exists():
        print(f"{args.split}: cache already in {output}, skipped", flush=True)
        return
    build_cache(args.split, checkpoint, output)


if __name__ == "__main__":
    main()
