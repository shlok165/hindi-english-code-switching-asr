"""Prove the single-file inference path matches the evaluated results.

    python -m src.check_inference --checkpoint data/checkpoints/baseline_rtx3050/best

Transcribes validation utterances one at a time from raw audio, the way a user
would, and compares each one with the cached log-probabilities that
src.evaluate decoded. The cache stores float16, so two characters within about
0.002 of each other in float32 can round to the same value and swap places in
greedy output. The check therefore fails when:
- any log-probability differs by more than float16 rounding, or
- a transcript differs anywhere other than at such a tie.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src import ROOT
from src.beam_search import LogProbCache
from src.data.preprocessing import load_audio
from src.inference import log_probabilities, transcribe_audio
from src.logit_cache import LOWEST_LOG_PROB, cache_directory
from src.models.ctc_model import load_model
from src.utils.config import data_path

# float16 steps are 2**-6 between 16 and 32, so rounding moves a value by at most
# half that. The extra 0.001 covers run-to-run GPU differences.
ROUNDING = 2**-7 + 1e-3


def is_tie(fresh: np.ndarray, frame: int) -> bool:
    """True when the two best characters are closer than float16 can separate."""
    first, second = np.sort(fresh[frame])[-2:][::-1]
    return first - second <= 2 * ROUNDING


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--utterances", type=int, default=25)
    args = parser.parse_args()
    checkpoint = data_path(args.checkpoint)

    path = ROOT / "results" / "evaluations" / "val_greedy.jsonl"
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    chosen = rows[:: max(len(rows) // args.utterances, 1)][: args.utterances]
    cache = LogProbCache(cache_directory(checkpoint), "val")
    position = {row["id"]: number for number, row in enumerate(cache.index)}

    model, vocab, extractor = load_model(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    samples, exact, ties, failures = [], 0, 0, 0
    for row in chosen:
        text = transcribe_audio(data_path(row["audio"]), model, vocab, extractor, device, row["start"], row["end"])
        audio = load_audio(str(data_path(row["audio"])), row["start"], row["end"])
        fresh = log_probabilities(audio, model, extractor, device).clamp(min=LOWEST_LOG_PROB).cpu().numpy()
        cached = cache[position[row["id"]]].astype(np.float32)
        difference = float(np.abs(fresh - cached).max()) if fresh.shape == cached.shape else float("inf")
        changed = np.nonzero(fresh.argmax(1) != cached.argmax(1))[0] if fresh.shape == cached.shape else []
        if text == row["hypothesis_raw"]:
            outcome = "match"
            exact += 1
        elif difference <= ROUNDING and len(changed) and all(is_tie(fresh, f) for f in changed):
            outcome = "float16 tie"
            ties += 1
        else:
            outcome = "mismatch"
        if difference > ROUNDING:
            outcome = "mismatch"
        failures += outcome == "mismatch"
        samples.append({"id": row["id"], "audio": row["audio"], "start": row["start"], "end": row["end"],
                        "reference": row["reference"], "prediction": text,
                        "evaluated_prediction": row["hypothesis_raw"], "outcome": outcome,
                        "max_log_prob_difference": round(difference, 5)})
    report = {"utterances": len(samples), "exact_matches": exact, "float16_ties": ties,
              "mismatches": failures, "tolerance": ROUNDING, "samples": samples}
    output = ROOT / "results" / "inference_check.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("utterances", "exact_matches", "float16_ties", "mismatches")}))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
