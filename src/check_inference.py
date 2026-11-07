"""Prove the single-file inference path matches the evaluated results.

    python -m src.check_inference --checkpoint data/checkpoints/baseline_rtx3050/best

Transcribes validation utterances one at a time from raw audio, the way a user
would, and compares each result with the greedy file from src.evaluate. Any
mismatch means the two paths disagree, and the command fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from src import ROOT
from src.inference import transcribe_audio
from src.models.ctc_model import load_model
from src.utils.config import data_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--utterances", type=int, default=25)
    args = parser.parse_args()

    path = ROOT / "results" / "evaluations" / "val_greedy.jsonl"
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    chosen = rows[:: max(len(rows) // args.utterances, 1)][: args.utterances]

    model, vocab, extractor = load_model(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    samples, mismatches = [], 0
    for row in chosen:
        text = transcribe_audio(data_path(row["audio"]), model, vocab, extractor, device, row["start"], row["end"])
        matched = text == row["hypothesis_raw"]
        mismatches += not matched
        samples.append({"id": row["id"], "audio": row["audio"], "start": row["start"], "end": row["end"],
                        "reference": row["reference"], "prediction": text, "matches_batch_result": matched})
    report = {"utterances": len(samples), "mismatches": mismatches, "samples": samples}
    output = ROOT / "results" / "inference_check.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"utterances": len(samples), "mismatches": mismatches}))
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
