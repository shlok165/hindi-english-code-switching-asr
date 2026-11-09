"""What the released val cuts cost a model trained on realigned segments (iteration 5).

    python -m src.analysis.cut_effects \
        --predictions iteration4=results/iteration4/evaluations iteration5=results/iteration5/evaluations \
        --checkpoints iteration4=data/checkpoints/iteration4/best iteration5=data/checkpoints/iteration5/best

Iteration 5 trains on realigned segments, but val, test and blind keep their
released cuts. Two checks separate the model from the cuts. Neither is used to
choose between iterations, which stays on val as released.

- positions: each word error in the evaluation's predictions, on every split,
  counted by where it falls: before the first correct word, between the first
  and the last, or after the last. Errors at the edges are the ones a cut can
  cause. Test and blind are reported here as in every error analysis, never used
  to choose.
- cuts: val realigned the way training was (src.data.realign, with the model that
  realigned training). Each released cut is classed by whether it holds only its
  own transcript's speech, and each model is decoded greedily, segment by
  segment, at the released and at the realigned cuts. The realigned cuts are
  placed using the val references, so their WER is not a result. It only
  measures what the released cuts cost.

Writes results/iteration5/cut_effects.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Iterable

import numpy as np

from src import ROOT
from src.metrics.wer import DEL, INS, MATCH, SUB, align
from src.utils.config import data_path, read_manifest

TOLERANCE = 0.2  # seconds of speech allowed on the wrong side of a released cut
STRAY = 2        # characters of untranscribed speech allowed inside one
PLACES = ("before the first correct word", "between the first and last correct words",
          "after the last correct word", "no correct word")
CLEAN, THROUGH = "released cut holds only its own speech", "released cut goes through speech"


def error_positions(pairs: Iterable[tuple[str, str]]) -> dict:
    """Errors by where they fall in each utterance, for (reference, hypothesis) pairs."""
    counts, words = Counter(), 0
    for reference, hypothesis in pairs:
        ref, hyp = reference.split(), hypothesis.split()
        if not ref:
            continue
        words += len(ref)
        ops = [op for op, _, _ in align(ref, hyp)]
        correct = [k for k, op in enumerate(ops) if op == MATCH]
        for k, op in enumerate(ops):
            if op == MATCH:
                continue
            if not correct:
                place = PLACES[3]
            elif k < correct[0]:
                place = PLACES[0]
            elif k > correct[-1]:
                place = PLACES[2]
            else:
                place = PLACES[1]
            counts[place, op] += 1
    result: dict = {"reference_words": words, "wer": round(100 * sum(counts.values()) / words, 2) if words else 0.0}
    for place in PLACES:
        ops = {name: counts[place, op] for name, op in (("substitutions", SUB), ("deletions", DEL), ("insertions", INS))}
        result[place] = {**ops, "wer_points": round(100 * sum(ops.values()) / words, 2) if words else 0.0}
    return result


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def classify(rows: list[dict], frames: np.ndarray, vocab: dict[str, int],
             tolerance: float = TOLERANCE, stray_limit: int = STRAY) -> dict | None:
    """Realigned cut and released-cut class for each segment of one recording."""
    from src.data.realign import FRAME_SECONDS, PAD_SECONDS, align_spans, segment_bounds, untranscribed_speech
    from src.models.tokenizer import BLANK, DELIMITER

    spans = align_spans(rows, frames, vocab)
    if spans is None:
        return None
    origin = rows[0]["start"]
    blank, delimiter = vocab[BLANK], vocab[DELIMITER]
    speech = untranscribed_speech(frames, *spans, blank, delimiter)
    starts, ends = segment_bounds(*spans, origin, origin, rows[-1]["end"], PAD_SECONDS, speech)
    first, last = spans
    own_start, own_end = origin + first * FRAME_SECONDS, origin + (last + 1) * FRAME_SECONDS
    best = frames.argmax(axis=1)
    covered = np.zeros(len(frames), dtype=bool)
    for f, l in zip(first, last):
        covered[f:l + 1] = True
    stray = np.concatenate(([0], np.cumsum((best != blank) & (best != delimiter) & ~covered)))
    found = {}
    for i, row in enumerate(rows):
        a, b = row["start"], row["end"]
        own_outside = (own_end[i] - own_start[i]) - overlap(a, b, own_start[i], own_end[i])
        other_inside = sum(overlap(a, b, own_start[j], own_end[j]) for j in range(len(rows)) if j != i)
        strays = int(stray[min(len(frames), round((b - origin) / FRAME_SECONDS))]
                     - stray[max(0, round((a - origin) / FRAME_SECONDS))])
        found[row["id"]] = {
            "realigned": (round(float(starts[i]), 3), round(float(ends[i]), 3)),
            "clean": bool(own_outside <= tolerance and other_inside <= tolerance and strays < stray_limit),
        }
    return found


def cut_check(val: list[dict], checkpoints: dict[str, str], align_with: str) -> dict:
    import torch

    from src.beam_search import greedy_from_log_probs
    from src.data.preprocessing import TARGET_SAMPLE_RATE, load_audio
    from src.data.realign import recording_frames, recordings
    from src.inference import log_probabilities
    from src.metrics.official import load_maps, official_wer
    from src.models.ctc_model import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grouped = recordings(val)

    def recording_audio(rows: list[dict]) -> np.ndarray:
        return load_audio(str(data_path(rows[0]["audio"])), rows[0]["start"], rows[-1]["end"])

    model, vocab, extractor = load_model(data_path(align_with))
    model.to(device).eval()
    segments, failed = {}, []
    for name, rows in grouped.items():
        found = classify(rows, recording_frames(recording_audio(rows), model, extractor, device), vocab)
        if found is None:
            failed.append(name)
        else:
            segments.update(found)

    hypotheses: dict[str, dict[str, dict[str, str]]] = {}
    for model_name, checkpoint in checkpoints.items():
        model, vocab, extractor = load_model(data_path(checkpoint))
        model.to(device).eval()
        hypotheses[model_name] = {"released": {}, "realigned": {}}
        for rows in grouped.values():
            origin, audio = rows[0]["start"], recording_audio(rows)
            for row in rows:
                if row["id"] not in segments:
                    continue
                for cut, (a, b) in (("released", (row["start"], row["end"])), ("realigned", segments[row["id"]]["realigned"])):
                    piece = audio[round((a - origin) * TARGET_SAMPLE_RATE):round((b - origin) * TARGET_SAMPLE_RATE)]
                    scores = log_probabilities(piece, model, extractor, device).cpu().numpy()
                    hypotheses[model_name][cut][row["id"]] = greedy_from_log_probs(scores, vocab)

    maps = load_maps()
    kept = [row for row in val if row["id"] in segments]
    groups = {"all": kept,
              CLEAN: [row for row in kept if segments[row["id"]]["clean"]],
              THROUGH: [row for row in kept if not segments[row["id"]]["clean"]]}
    result: dict = {"aligned_with": align_with, "recordings": len(grouped), "recordings_failed": failed,
                    "tolerance_seconds": TOLERANCE, "stray_characters": STRAY}
    for group, rows in groups.items():
        entry = {"segments": len(rows), "hours": round(sum(row["duration"] for row in rows) / 3600, 2)}
        for model_name in checkpoints:
            for cut in ("released", "realigned"):
                scored = official_wer([(row["text_raw"], hypotheses[model_name][cut][row["id"]]) for row in rows], maps)
                entry["reference_words"] = scored["reference_words"]
                entry[f"{model_name}, {cut} cuts, greedy WER"] = round(scored["wer"], 2)
        result[group] = entry
    return result


def named(values: list[str]) -> dict[str, str]:
    return dict(value.split("=", 1) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", nargs="+", required=True, help="NAME=evaluations directory")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="NAME=checkpoint")
    parser.add_argument("--align-with", default="data/checkpoints/iteration4/best")
    parser.add_argument("--manifest-dir", default="data/processed/manifests")
    parser.add_argument("--output", default="results/iteration5/cut_effects.json")
    args = parser.parse_args()

    val = read_manifest(data_path(args.manifest_dir) / "val.jsonl")
    if any(row["split"] != "val" for row in val):
        raise ValueError("the cut check reads val only")
    positions: dict = {}
    for name, directory in named(args.predictions).items():
        for split in ("val", "test", "blind"):
            for decoder in ("greedy", "beam"):
                with (ROOT / directory / f"{split}_{decoder}.jsonl").open(encoding="utf-8") as handle:
                    rows = [json.loads(line) for line in handle]
                positions.setdefault(split, {}).setdefault(decoder, {})[name] = error_positions(
                    (row["reference"], row["hypothesis"]) for row in rows)
    result = {"positions": positions, "cuts": cut_check(val, named(args.checkpoints), args.align_with)}
    (ROOT / args.output).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
