"""Re-cut the training segments where their transcripts actually change.

    python -m src.data.realign log-probs --checkpoint data/checkpoints/iteration4/best
    python -m src.data.realign align --checkpoint data/checkpoints/iteration4/best \
        --manifest-dir data/processed/manifests --output-dir data/processed/manifests_realigned

MUCS training segments tile each recording with no gaps, but some cut points sit
seconds away from where the transcript changes, so a segment's audio holds part of
its neighbour's speech or misses part of its own. Iteration 4 drops the worst of
them. This finds the cut points again instead, in two steps, on the training split
only:

1. ``log-probs`` scores every training recording, from its first segment's start to
   its last segment's end, with one model, in overlapping 30 s windows so that
   every frame has context on both sides. The frames are stored once per recording.
2. ``align`` force-aligns each recording's transcripts, in order, to those frames
   with CTC, allowing each transcript only within MARGIN_SECONDS of its original
   segment. Each cut point then moves to the middle of the pause between the last
   character of one transcript and the first character of the next. The segments
   still tile the recording, so no audio is removed or truncated. The misalignment
   filter of iteration 4 then runs on the new segments, scored greedily from the
   same frames, and drops those that still do not match.

Val, test and blind are copied unchanged. Val holds whole recordings, so a moved
cut point can never reach val audio.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from src import ROOT
from src.beam_search import greedy_from_log_probs
from src.data.dataset import SAMPLES_PER_FRAME
from src.data.filter_train import MAX_CER, Skeleton, error_rates
from src.data.preprocessing import TARGET_SAMPLE_RATE, load_audio
from src.data.script_normalisation import load_cmudict
from src.models.tokenizer import BLANK, DELIMITER, encode
from src.utils.config import data_path, read_manifest

FRAME_SECONDS = SAMPLES_PER_FRAME / TARGET_SAMPLE_RATE  # 20 ms
WINDOW_SAMPLES = 30 * TARGET_SAMPLE_RATE
HOP_SAMPLES = 25 * TARGET_SAMPLE_RATE  # a multiple of SAMPLES_PER_FRAME, so windows share one frame grid
LOWEST_LOG_PROB = -30.0  # as in src.logit_cache
MARGIN_SECONDS = 15.0
MOVE_BANDS = (0.1, 0.5, 1.0, 2.0, 5.0)
EXAMPLES = 12


def frames_directory(checkpoint: Path) -> Path:
    return data_path("data/processed/realign") / checkpoint.parent.name / checkpoint.name


def recordings(entries: list[dict]) -> dict[str, list[dict]]:
    """Training segments grouped by recording, in time order."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        grouped[entry["recording"]].append(entry)
    return {name: sorted(rows, key=lambda row: row["start"]) for name, rows in sorted(grouped.items())}


# --- step 1: frames for whole recordings -------------------------------------------

def window_starts(samples: int, window: int = WINDOW_SAMPLES, hop: int = HOP_SAMPLES) -> list[int]:
    """Starts every `hop` samples; the last window still holds at least window - hop samples."""
    if samples <= window:
        return [0]
    return [k * hop for k in range(math.ceil((samples - (window - hop)) / hop))]


def stitch(window_scores: list[np.ndarray], window: int = WINDOW_SAMPLES, hop: int = HOP_SAMPLES) -> np.ndarray:
    """Join per-window frames, keeping the middle of each overlap from each side."""
    hop_frames = hop // SAMPLES_PER_FRAME
    trim = (window - hop) // 2 // SAMPLES_PER_FRAME
    parts = []
    for k, scores in enumerate(window_scores):
        first = 0 if k == 0 else trim
        last = len(scores) if k == len(window_scores) - 1 else hop_frames + trim
        parts.append(scores[first:last])
    return np.concatenate(parts)


def recording_frames(audio: np.ndarray, model, extractor, device) -> np.ndarray:
    from src.inference import log_probabilities  # imports torch and the model code

    scores = [log_probabilities(audio[start:start + WINDOW_SAMPLES], model, extractor, device)
              .clamp(min=LOWEST_LOG_PROB).cpu().numpy().astype(np.float16)
              for start in window_starts(len(audio))]
    frames = stitch(scores)
    import torch

    expected = int(model._get_feat_extract_output_lengths(torch.tensor(len(audio))))
    if len(frames) != expected:
        raise RuntimeError(f"stitched {len(frames)} frames, expected {expected}")
    return frames


def write_frames(checkpoint: Path, manifest_dir: Path) -> None:
    import torch

    from src.models.ctc_model import load_model

    output = frames_directory(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    grouped = recordings(read_manifest(manifest_dir / "train.jsonl"))
    model, _, extractor = load_model(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    began, done = time.time(), 0
    for done, (name, rows) in enumerate(grouped.items(), 1):
        index = output / f"{name}.json"
        if index.exists() and (output / f"{name}.npy").exists():
            continue
        start, end = rows[0]["start"], rows[-1]["end"]
        audio = load_audio(str(data_path(rows[0]["audio"])), start, end)
        frames = recording_frames(audio, model, extractor, device)
        np.save(output / f"{name}.npy", frames)
        # Written after the array, so an index means the frames are complete.
        index.write_text(json.dumps({"start": start, "end": end, "frames": len(frames)}), encoding="utf-8")
        if done % 25 == 0:
            print(f"{done}/{len(grouped)} recordings, {time.time() - began:.0f} s", flush=True)
    print(f"frames for {done} recordings in {output}, {time.time() - began:.0f} s", flush=True)


# --- step 2: alignment --------------------------------------------------------------

def ctc_states(transcripts: list[list[int]], blank: int, delimiter: int) -> tuple[np.ndarray, np.ndarray]:
    """CTC states for transcripts in order, joined by word delimiters, and each label's owner.

    States alternate blank, label, blank, ... Owner is the transcript's index for its own
    labels and -1 for the delimiters that join transcripts.
    """
    labels, owners = [], []
    for index, transcript in enumerate(transcripts):
        if not transcript:
            raise ValueError(f"transcript {index} has no labels")
        if index:
            labels.append(delimiter)
            owners.append(-1)
        labels.extend(transcript)
        owners.extend([index] * len(transcript))
    states = np.full(2 * len(labels) + 1, blank, dtype=np.int64)
    states[1::2] = labels
    return states, np.array(owners)


def state_bounds(owners: np.ndarray, first: np.ndarray, last: np.ndarray, frames: int) -> tuple[np.ndarray, np.ndarray]:
    """First and last frame at which each state may be visited.

    A label may be visited within its transcript's frame range, a joining delimiter
    within either neighbour's, and a blank within either neighbouring label's.
    Both bounds rise with the state index, so each frame allows a contiguous range.
    """
    joins = np.flatnonzero(owners < 0)
    previous, following = owners.copy(), owners.copy()
    previous[joins], following[joins] = owners[joins - 1], owners[joins + 1]
    label_first, label_last = first[previous], last[following]
    lower = np.empty(2 * len(owners) + 1, dtype=np.int64)
    upper = np.empty_like(lower)
    lower[1::2], upper[1::2] = label_first, label_last
    lower[0], upper[0] = 0, label_last[0]
    lower[2:-1:2], upper[2:-1:2] = label_first[:-1], label_last[1:]
    lower[-1], upper[-1] = label_first[-1], frames - 1
    return np.maximum.accumulate(lower), np.maximum.accumulate(upper)


def banded_viterbi(log_probs: np.ndarray, states: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray | None:
    """Best CTC path, as the state of each frame, keeping each state inside its frame bounds.

    Returns None when no path fits the bounds.
    """
    frames, count = len(log_probs), len(states)
    times = np.arange(frames)
    low = np.searchsorted(upper, times, side="left")
    high = np.searchsorted(lower, times, side="right") - 1
    if frames < 1 or low[0] > 1 or high[0] < 0 or high[-1] < count - 1 or np.any(low > high):
        return None
    skip = np.zeros(count, dtype=bool)
    skip[3::2] = states[3::2] != states[1:-2:2]  # a label may follow the previous label directly
    # score[s + 2] holds state s; the two leading cells stay -inf for the shifted reads.
    score = np.full(count + 2, -np.inf)
    first_states = np.arange(low[0], min(high[0], 1) + 1)
    score[first_states + 2] = log_probs[0, states[first_states]]
    choices: list[np.ndarray] = []
    for t in range(1, frames):
        a, b = low[t], high[t]
        two = score[a:b + 1].copy()
        two[~skip[a:b + 1]] = -np.inf
        candidates = np.stack((score[a + 2:b + 3], score[a + 1:b + 2], two))
        choice, best = candidates.argmax(axis=0), candidates.max(axis=0)
        score[low[t - 1] + 2:a + 2] = -np.inf  # states that fell out of the band
        score[a + 2:b + 3] = best + log_probs[t, states[a:b + 1]]
        choices.append(choice.astype(np.int8))
    ends = [s for s in (count - 1, count - 2) if s >= low[-1]]
    state = max(ends, key=lambda s: score[s + 2])
    if not np.isfinite(score[state + 2]):
        return None
    path = np.empty(frames, dtype=np.int64)
    for t in range(frames - 1, 0, -1):
        path[t] = state
        state -= int(choices[t - 1][state - low[t]])
    path[0] = state
    return path if state <= 1 else None


def transcript_spans(path: np.ndarray, owners: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """First and last frame at which each transcript emits one of its own labels."""
    emitting = path % 2 == 1
    owner = np.full(len(path), -1)
    owner[emitting] = owners[(path[emitting] - 1) // 2]
    first, last = np.full(count, len(path)), np.full(count, -1)
    mine = owner >= 0
    np.minimum.at(first, owner[mine], np.flatnonzero(mine))
    np.maximum.at(last, owner[mine], np.flatnonzero(mine))
    return first, last


def cut_points(first: np.ndarray, last: np.ndarray) -> np.ndarray:
    """Frame of each cut between neighbours: the middle of the pause between them."""
    return (last[:-1] + 1 + first[1:]) / 2


def realign_recording(rows: list[dict], frames: np.ndarray, vocab: dict[str, int],
                      margin: float = MARGIN_SECONDS) -> list[tuple[float, float]] | None:
    """New (start, end) for each segment of one recording, or None if no alignment fits."""
    origin = rows[0]["start"]
    total = len(frames)
    # The tolerance keeps times that land on a frame edge, such as 0.08 / 0.02, on that edge.
    first = np.array([max(0, math.floor((row["start"] - origin - margin) / FRAME_SECONDS + 1e-6)) for row in rows])
    last = np.array([min(total - 1, math.ceil((row["end"] - origin + margin) / FRAME_SECONDS - 1e-6)) for row in rows])
    states, owners = ctc_states([encode(row["text"], vocab) for row in rows], vocab[BLANK], vocab[DELIMITER])
    lower, upper = state_bounds(owners, first, last, total)
    path = banded_viterbi(frames.astype(np.float32), states, lower, upper)
    if path is None:
        return None
    spans = transcript_spans(path, owners, len(rows))
    cuts = origin + cut_points(*spans) * FRAME_SECONDS
    starts = [rows[0]["start"], *cuts.tolist()]
    ends = [*cuts.tolist(), rows[-1]["end"]]
    return [(round(s, 3), round(e, 3)) for s, e in zip(starts, ends)]


def frame_range(start: float, end: float, origin: float, total: int) -> slice:
    return slice(max(0, round((start - origin) / FRAME_SECONDS)), min(total, round((end - origin) / FRAME_SECONDS)))


def align(checkpoint: Path, manifest_dir: Path, output_dir: Path, report_path: Path,
          margin: float, max_cer: float, cmudict: Path, names: dict[str, str]) -> None:
    source = frames_directory(checkpoint)
    vocab = json.loads((checkpoint / "vocab.json").read_text(encoding="utf-8"))
    train = read_manifest(manifest_dir / "train.jsonl")
    if any(row.get("split", "train") != "train" for row in train):
        raise ValueError("only training rows can be realigned")
    skeleton = Skeleton(load_cmudict(cmudict))
    began = time.time()
    scored, failed = {}, []
    for done, (name, rows) in enumerate(recordings(train).items(), 1):
        index = json.loads((source / f"{name}.json").read_text(encoding="utf-8"))
        frames = np.load(source / f"{name}.npy")
        if abs(index["start"] - rows[0]["start"]) > 1e-6 or abs(index["end"] - rows[-1]["end"]) > 1e-6:
            raise ValueError(f"{name}: frames were written for other segments")
        new = realign_recording(rows, frames, vocab, margin)
        if new is None:
            failed.append(name)
            new = [(row["start"], row["end"]) for row in rows]
        for row, (start, end) in zip(rows, new):
            reference = row.get("text_unnormalised", row["text"])
            before = greedy_from_log_probs(frames[frame_range(row["start"], row["end"], rows[0]["start"], len(frames))], vocab)
            after = greedy_from_log_probs(frames[frame_range(start, end, rows[0]["start"], len(frames))], vocab)
            scored[row["id"]] = {
                "id": row["id"], "recording": name, "reference": reference,
                "old": [row["start"], row["end"]], "new": [start, end],
                "hypothesis_before": before, "hypothesis_after": after,
                "cer_before": [round(x, 2) for x in error_rates(reference, before, skeleton)],
                "cer_after": [round(x, 2) for x in error_rates(reference, after, skeleton)],
                "labels": len(encode(row["text"], vocab)),
            }
        if done % 50 == 0:
            print(f"aligned {done} recordings, {time.time() - began:.0f} s", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    kept_rows, dropped = [], set()
    for row in train:
        item = scored[row["id"]]
        if min(item["cer_after"]) > max_cer:
            dropped.add(row["id"])
            continue
        start, end = item["new"]
        kept_rows.append({**row, "start": start, "end": end, "duration": round(end - start, 3)})
    with (output_dir / "train.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in kept_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for split in ("val", "test", "blind"):
        shutil.copy2(manifest_dir / f"{split}.jsonl", output_dir / f"{split}.jsonl")

    report = summarise(train, scored, dropped, failed, kept_rows, names, margin, max_cer)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("recordings_failed", "cut_points_moved", "filter_before",
                                                    "filter_after", "kept_hours")}, ensure_ascii=False))


def summarise(train, scored, dropped, failed, kept_rows, names, margin, max_cer) -> dict:
    ordered = [scored[row["id"]] for row in train]
    moves = []
    for name, rows in recordings(train).items():
        for row in rows[1:]:
            item = scored[row["id"]]
            moves.append(abs(item["new"][0] - item["old"][0]))
    moves = np.array(moves)
    hours = sum(row["duration"] for row in train) / 3600
    before = [item for item in ordered if min(item["cer_before"]) > max_cer]
    previously = {item["id"] for item in before}
    rescued = [item for item in before if item["id"] not in dropped]
    newly = [item for item in ordered if item["id"] in dropped and item["id"] not in previously]
    durations = np.array([row["duration"] for row in kept_rows])
    sample = random.Random(0)
    fields = ("id", "old", "new", "cer_before", "cer_after", "reference", "hypothesis_before", "hypothesis_after")
    largest = sorted(ordered, key=lambda item: -abs(item["new"][0] - item["old"][0]))[:EXAMPLES]
    return {
        "rule": "force-align each training recording's transcripts in order with CTC, within "
                f"{margin} s of their original segments; cut in the middle of the pause between "
                f"transcripts; drop a segment whose CER is still above {max_cer} as written and in one script",
        **names, "margin_seconds": margin, "max_cer": max_cer,
        "recordings": len(recordings(train)), "recordings_failed": failed,
        "segments": len(train), "train_hours": round(hours, 2),
        "cut_points": len(moves),
        "cut_points_moved": {f"over {band} s": int((moves > band).sum()) for band in MOVE_BANDS},
        "median_move_seconds": round(float(np.median(moves)), 3) if len(moves) else 0.0,
        "filter_before": {"dropped": len(before),
                          "hours": round(sum(i["old"][1] - i["old"][0] for i in before) / 3600, 2)},
        "filter_after": {"dropped": len(dropped),
                         "hours": round(sum(scored[i]["new"][1] - scored[i]["new"][0] for i in dropped) / 3600, 2)},
        "rescued_by_realignment": len(rescued),
        "newly_dropped": len(newly),
        "kept_segments": len(kept_rows), "kept_hours": round(float(durations.sum()) / 3600, 2),
        "kept_outside_duration_limits": int(((durations < 0.5) | (durations > 60)).sum()),
        "kept_labels_exceed_frames": int(sum(scored[row["id"]]["labels"] > round(row["duration"] * TARGET_SAMPLE_RATE)
                                             // SAMPLES_PER_FRAME for row in kept_rows)),
        "examples_rescued": [{k: i[k] for k in fields} for i in sample.sample(rescued, min(EXAMPLES, len(rescued)))],
        "examples_newly_dropped": [{k: i[k] for k in fields} for i in sample.sample(newly, min(EXAMPLES, len(newly)))],
        "examples_largest_moves": [{k: i[k] for k in fields} for i in largest],
        "dropped_ids": sorted(dropped),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("log-probs", "align"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest-dir", default="data/processed/manifests")
    parser.add_argument("--output-dir", default="data/processed/manifests_realigned")
    parser.add_argument("--report", default="results/train_realign_report.json")
    parser.add_argument("--margin", type=float, default=MARGIN_SECONDS)
    parser.add_argument("--max-cer", type=float, default=MAX_CER)
    parser.add_argument("--cmudict", default="data/external/cmudict/cmudict.dict")
    args = parser.parse_args()
    checkpoint, manifests = data_path(args.checkpoint), data_path(args.manifest_dir)
    if args.step == "log-probs":
        write_frames(checkpoint, manifests)
    else:
        names = {"checkpoint": args.checkpoint, "manifests": args.manifest_dir, "output": args.output_dir}
        align(checkpoint, manifests, data_path(args.output_dir), ROOT / args.report,
              args.margin, args.max_cer, data_path(args.cmudict), names)


if __name__ == "__main__":
    main()
