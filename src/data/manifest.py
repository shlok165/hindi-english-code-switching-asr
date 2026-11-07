"""Turn the extracted Kaldi folders into manifests, and audit the dataset.

    python -m src.data.manifest --config configs/data.yaml

Writes train/val/test/blind JSONL manifests and docs/dataset_stats.json. Every
figure in that file is measured from the data on disk. Official splits are kept
apart; val is carved out of train by whole recording, so one tutorial's audio
never lands on both sides. No speaker labels exist, so val is not speaker-disjoint.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from src import ROOT
from src.data.preprocessing import TARGET_SAMPLE_RATE, clean_transcript
from src.utils.config import data_path, load_config

LATIN = re.compile("[a-z]")
DEVANAGARI = re.compile("[ऀ-ॿ]")
SPLIT_FOLDERS = {
    "train": ("train", "train"),
    "test": ("test", "test"),
    "blind": ("blind", "subtask2_blindtest_wReadme/hindi"),
}


def portable_path(path: Path, extracted: Path) -> str:
    """A data/... path, so a manifest works wherever MUCS_DATA_ROOT points."""
    return "data/" + path.relative_to(extracted.parents[1]).as_posix()


def read_table(path: Path) -> dict[str, str]:
    """Read a Kaldi table: key, then the rest of the line."""
    table = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.strip().partition(" ")
        if key:
            table[key] = value.strip()
    return table


def locate_kaldi_dir(folder: Path) -> Path:
    candidates = sorted(p.parent for p in folder.rglob("segments"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one Kaldi folder under {folder}, found {candidates}")
    return candidates[0]


def inspect_recording(path: Path) -> dict[str, Any]:
    """Read a whole file, so truncated and corrupt audio is actually caught."""
    try:
        with sf.SoundFile(path) as handle:
            info = {"rate": handle.samplerate, "channels": handle.channels,
                    "subtype": handle.subtype, "frames": handle.frames}
            audio = handle.read(dtype="float32")
    except Exception as error:  # soundfile raises several types on damaged files
        return {"error": f"{type(error).__name__}: {error}"}
    info["seconds"] = info["frames"] / info["rate"]
    info["complete"] = len(audio) == info["frames"]
    info["silent"] = bool(np.abs(audio).max(initial=0) == 0)
    info["non_finite"] = not bool(np.isfinite(audio).all())
    return info


def build_split(split: str, extracted: Path) -> tuple[list[dict], dict[str, Any]]:
    top, inner = SPLIT_FOLDERS[split]
    folder = extracted / top / inner
    kaldi = locate_kaldi_dir(folder)
    texts = read_table(kaldi / "text")
    segments = read_table(kaldi / "segments")
    scp = read_table(kaldi / "wav.scp")
    wavs = {p.stem: p for p in folder.rglob("*.wav")}

    recordings = sorted({value.split()[0] for value in segments.values()})
    paths = {r: wavs.get(r) for r in recordings}
    missing_audio = sorted(r for r, p in paths.items() if p is None)
    present = [paths[r] for r in recordings if paths[r] is not None]
    with ThreadPoolExecutor(8) as pool:
        details = dict(zip((p.stem for p in present), pool.map(inspect_recording, present)))

    unreadable = sorted(r for r, d in details.items() if "error" in d)
    entries, problems = [], Counter()
    for utterance, value in sorted(segments.items()):
        recording, start, end = value.split()[:3]
        start, end = float(start), float(end)
        detail = details.get(recording)
        if utterance not in texts:
            problems["segment_without_text"] += 1
        if detail is None or "error" in detail:
            problems["segment_without_audio"] += 1
            continue
        if end <= start:
            problems["non_positive_duration"] += 1
            continue
        if end > detail["seconds"] + 0.05:
            problems["segment_past_end_of_file"] += 1
            end = detail["seconds"]
            if end <= start:
                continue
        raw = texts.get(utterance, "")
        entries.append({
            "id": utterance,
            "split": split,
            "recording": recording,
            "audio": portable_path(paths[recording], extracted),
            "start": start,
            "end": end,
            "duration": round(end - start, 3),
            "text_raw": raw,
            "text": clean_transcript(raw),
        })
    problems["text_without_segment"] = len(set(texts) - set(segments))
    readable = [d for d in details.values() if "error" not in d]
    audit = {
        "recordings_listed": len(recordings),
        "wav_scp_entries": len(scp),
        "recordings_missing_audio": missing_audio,
        "recordings_unreadable": unreadable,
        "recordings_truncated": sorted(r for r, d in details.items() if d.get("complete") is False),
        "recordings_silent": sorted(r for r, d in details.items() if d.get("silent")),
        "recordings_non_finite": sorted(r for r, d in details.items() if d.get("non_finite")),
        "sample_rates_hz": dict(Counter(str(d["rate"]) for d in readable)),
        "channels": dict(Counter(str(d["channels"]) for d in readable)),
        "subtypes": dict(Counter(d["subtype"] for d in readable)),
        "recording_hours": round(sum(d["seconds"] for d in readable) / 3600, 2),
        "problems": dict(problems),
    }
    return entries, audit


def summarise(entries: list[dict], train_sentences: set[str] | None) -> dict[str, Any]:
    durations = np.array([e["duration"] for e in entries]) if entries else np.zeros(1)
    voiced = [e for e in entries if e["text"]]
    words = [w for e in voiced for w in e["text"].split()]
    latin = sum(bool(LATIN.search(w)) for w in words)
    devanagari = sum(bool(DEVANAGARI.search(w)) for w in words)
    rates = np.array([len(e["text"].replace(" ", "")) / max(e["duration"], 1e-3) for e in voiced] or [0.0])
    sentences = [e["text"] for e in voiced]
    summary = {
        "utterances": len(entries),
        "empty_transcripts": len(entries) - len(voiced),
        "unique_sentences": len(set(sentences)),
        "recordings": len({e["recording"] for e in entries}),
        "segment_hours": round(float(durations.sum()) / 3600, 2),
        "duration_seconds": {
            name: round(float(np.percentile(durations, q)), 2)
            for name, q in (("min", 0), ("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))
        },
        "seconds_over_20s": round(float(durations[durations > 20].sum()), 1),
        "utterances_over_20s": int((durations > 20).sum()),
        "utterances_under_0.5s": int((durations < 0.5).sum()),
        "words": len(words),
        "vocabulary_words": len(set(words)),
        "latin_word_percent": round(100 * latin / max(len(words), 1), 2),
        "devanagari_word_percent": round(100 * devanagari / max(len(words), 1), 2),
        "chars_per_second": {
            "p1": round(float(np.percentile(rates, 1)), 2),
            "p50": round(float(np.percentile(rates, 50)), 2),
            "p99": round(float(np.percentile(rates, 99)), 2),
        },
        "implausible_rate_over_35cps": int((rates > 35).sum()),
    }
    if train_sentences is not None:
        repeated = sum(s in train_sentences for s in sentences)
        summary["utterances_with_sentence_in_train_percent"] = round(100 * repeated / max(len(sentences), 1), 2)
    return summary


def split_validation(entries: list[dict], hours: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Hold out whole recordings until `hours` of speech are reserved."""
    seconds: Counter[str] = Counter()
    for entry in entries:
        seconds[entry["recording"]] += entry["duration"]
    order = sorted(seconds)
    random.Random(seed).shuffle(order)
    held, total = set(), 0.0
    for recording in order:
        if total >= hours * 3600:
            break
        held.add(recording)
        total += seconds[recording]
    train = [{**e} for e in entries if e["recording"] not in held]
    val = [{**e, "split": "val"} for e in entries if e["recording"] in held]
    return train, val


def write_manifest(entries: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    data = load_config(args.config)["data"]
    extracted = data_path(data["extracted_dir"])
    manifest_dir = data_path(data["manifest_dir"])
    manifest_dir.mkdir(parents=True, exist_ok=True)

    built = {split: build_split(split, extracted) for split in SPLIT_FOLDERS}
    train, val = split_validation(built["train"][0], data["val_hours"], data["val_seed"])
    train_sentences = {e["text"] for e in train if e["text"]}
    splits = {"train": train, "val": val, "test": built["test"][0], "blind": built["blind"][0]}
    for name, entries in splits.items():
        write_manifest(entries, manifest_dir / f"{name}.jsonl")

    overlap = {e["id"] for e in train} & {e["id"] for e in val}
    report = {
        "expected_sample_rate_hz": TARGET_SAMPLE_RATE,
        "audit": {split: audit for split, (_, audit) in built.items()},
        "splits": {
            name: summarise(entries, None if name == "train" else train_sentences)
            for name, entries in splits.items()
        },
        "train_val_shared_utterances": len(overlap),
        "train_val_shared_recordings": len({e["recording"] for e in train} & {e["recording"] for e in val}),
    }
    path = ROOT / "docs" / "dataset_stats.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
