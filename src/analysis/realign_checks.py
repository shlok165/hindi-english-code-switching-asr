"""The two measurements behind the realignment's settings (iteration 5).

    python -m src.analysis.realign_checks --checkpoint data/checkpoints/iteration4/best

Both use training recordings only, with the model that scored them for realignment.

- windows: over 24 two-minute stretches, the characters the model emits per
  transcript character when it decodes each segment alone, or whole stretches in
  overlapping windows of several lengths. Too long a window makes the model stop
  transcribing partway through; decoding segments alone splits the words that
  cross a cut.
- cuts: over 30 recordings, each segment decoded alone, as training sees it, at
  the released cuts and at realigned cuts under several rules. Segments the model
  never trained on (those iteration 4 dropped) show the effect without the model
  having memorised the released cuts.

Writes results/realign_checks.json.
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch

from src import ROOT
from src.beam_search import greedy_from_log_probs
from src.data.filter_train import MAX_CER, Skeleton, error_rates
from src.data.preprocessing import TARGET_SAMPLE_RATE, load_audio
from src.data.realign import (PAD_SECONDS, align_spans, frames_directory, recordings, segment_bounds, stitch,
                              untranscribed_speech, window_starts)
from src.data.script_normalisation import load_cmudict
from src.inference import log_probabilities
from src.models.ctc_model import load_model
from src.models.tokenizer import BLANK, DELIMITER
from src.utils.config import data_path, read_manifest

WINDOWS = ((12, 8), (16, 12), (20, 16), (30, 25))  # seconds: window, hop
# Setting: (padding in seconds, applied only where a gap holds untranscribed speech).
CUT_RULES = {
    "midpoint": (None, False),
    "0.15 s padding everywhere": (0.15, False),
    "0.3 s padding everywhere": (0.3, False),
    "0.5 s padding everywhere": (0.5, False),
    f"{PAD_SECONDS} s padding where a gap holds speech (used)": (PAD_SECONDS, True),
}


def characters(text: str) -> int:
    return len(text.replace(" ", ""))


def check_windows(grouped, model, extractor, vocab, device) -> dict:
    rng = random.Random(3)
    emitted = {"segments alone": [], **{f"{w} s windows every {h} s": [] for w, h in WINDOWS}}
    reference = []
    for name in rng.sample(sorted(grouped), 24):
        rows = grouped[name]
        first = rng.randrange(0, max(1, len(rows) - 20))
        stretch = rows[first:first + 20]
        start = stretch[0]["start"]
        audio = load_audio(str(data_path(rows[0]["audio"])), start, stretch[-1]["end"])
        reference.append(sum(characters(row["text"]) for row in stretch))
        alone = [audio[round((r["start"] - start) * TARGET_SAMPLE_RATE):round((r["end"] - start) * TARGET_SAMPLE_RATE)]
                 for r in stretch]
        emitted["segments alone"].append(sum(characters(greedy_from_log_probs(
            log_probabilities(piece, model, extractor, device).cpu().numpy(), vocab)) for piece in alone))
        for w, h in WINDOWS:
            window, hop = w * TARGET_SAMPLE_RATE, h * TARGET_SAMPLE_RATE
            scores = [log_probabilities(audio[s:s + window], model, extractor, device).cpu().numpy()
                      for s in window_starts(len(audio), window, hop)]
            emitted[f"{w} s windows every {h} s"].append(characters(greedy_from_log_probs(stitch(scores, window, hop), vocab)))
    reference = np.array(reference)
    result = {"stretches": len(reference), "transcript_characters": int(reference.sum())}
    for method, counts in emitted.items():
        ratios = np.array(counts) / reference
        result[method] = {"characters_per_transcript_character": round(float(np.sum(counts) / reference.sum()), 3),
                          "lowest_stretch": round(float(ratios.min()), 2),
                          "stretches_below_0.8": int((ratios < 0.8).sum())}
    return result


def check_cuts(grouped, model, extractor, vocab, device, checkpoint, dropped_before, skeleton) -> dict:
    source = frames_directory(checkpoint)
    settings = {"released": None, **CUT_RULES}
    found = {name: {"cer": [], "unseen": [], "seconds": 0.0} for name in settings}
    for name in random.Random(5).sample(sorted(grouped), 30):
        rows = grouped[name]
        frames = np.load(source / f"{name}.npy")
        spans = align_spans(rows, frames, vocab)
        origin, end = rows[0]["start"], rows[-1]["end"]
        audio = load_audio(str(data_path(rows[0]["audio"])), origin, end)
        speech = untranscribed_speech(frames, *spans, vocab[BLANK], vocab[DELIMITER])
        for setting, rule in settings.items():
            if rule is None:
                bounds = [(row["start"], row["end"]) for row in rows]
            else:
                pad, speech_only = rule
                bounds = list(zip(*segment_bounds(*spans, origin, origin, end, 1e9 if pad is None else pad,
                                                  speech if speech_only else None)))
            for row, (s, e) in zip(rows, bounds):
                piece = audio[round((s - origin) * TARGET_SAMPLE_RATE):round((e - origin) * TARGET_SAMPLE_RATE)]
                text = greedy_from_log_probs(log_probabilities(piece, model, extractor, device).cpu().numpy(), vocab)
                cer = min(error_rates(row["text"], text, skeleton))
                found[setting]["cer"].append(cer)
                found[setting]["seconds"] += e - s
                if row["id"] in dropped_before:
                    found[setting]["unseen"].append(cer)
    result = {"recordings": 30, "segments": len(found["released"]["cer"]),
              "segments_never_trained_on": len(found["released"]["unseen"]), "pad_seconds": PAD_SECONDS}
    for setting, values in found.items():
        cer, unseen = np.array(values["cer"]), np.array(values["unseen"])
        result[setting] = {"median_cer": round(float(np.median(cer)), 1), "mean_cer": round(float(cer.mean()), 1),
                           f"above_{MAX_CER:g}": int((cer > MAX_CER).sum()), "audio_hours": round(values["seconds"] / 3600, 2),
                           "never_trained_on_median_cer": round(float(np.median(unseen)), 1),
                           f"never_trained_on_above_{MAX_CER:g}": int((unseen > MAX_CER).sum())}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest-dir", default="data/processed/manifests")
    parser.add_argument("--dropped-before", default="results/train_filter_report.json")
    parser.add_argument("--cmudict", default="data/external/cmudict/cmudict.dict")
    parser.add_argument("--output", default="results/realign_checks.json")
    args = parser.parse_args()
    checkpoint = data_path(args.checkpoint)
    model, vocab, extractor = load_model(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    grouped = recordings(read_manifest(data_path(args.manifest_dir) / "train.jsonl"))
    dropped_before = set(json.loads((ROOT / args.dropped_before).read_text(encoding="utf-8"))["dropped_ids"])
    skeleton = Skeleton(load_cmudict(data_path(args.cmudict)))
    result = {"checkpoint": args.checkpoint,
              "windows": check_windows(grouped, model, extractor, vocab, device),
              "cuts": check_cuts(grouped, model, extractor, vocab, device, checkpoint, dropped_before, skeleton)}
    (ROOT / args.output).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
