"""Drop training utterances whose transcript does not match their audio (iteration 4).

    python -m src.data.filter_train --manifest-dir data/processed/manifests_latin \
        --output-dir data/processed/manifests_latin_clean

MUCS segments are cut from long recordings, and some cuts leave the transcript
holding words the audio lacks, or the other way round. Iteration 1's model,
scored on its own training data by src.analysis.score_train, still gets most
characters of those utterances wrong after six epochs on them, against a median
of 4.6 % on the rest. An utterance is dropped when its character error rate is
above MAX_CER in both of two views:

- as written, and
- in one script: each side reduced to a consonant skeleton, with English words
  first spelled in Devanagari through CMUdict, as for the script map, then
  vowels, halant, nukta, aspiration and dental against retroflex removed.

The second view keeps utterances that differ only in script, such as कंसोल विंडो
against "console window". The first keeps those whose English words CMUdict
lacks or misspells, such as "acetate" against "acitate". References are always
the original transcripts, so the same utterances are dropped whichever
manifests are filtered.

Only the training split is filtered. Val, test and blind are copied unchanged,
so iterations stay comparable. MAX_CER was set by reading random training
utterances on either side of it, before any model was trained on the result.
Just above it, nearly every utterance has a transcript that the audio does not
hold: half of it cut off, a different sentence, or repeated words. Just below
it most are still imperfect, but usually only by a word or two at an edge,
mixed with speech the model simply got wrong. This is a data-quality filter,
not a way to fit the GPU: no utterance is dropped for its length.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import unicodedata
from pathlib import Path

import numpy as np

from src import ROOT
from src.data.script_normalisation import load_cmudict, pronunciations, spellings
from src.metrics.cer import score as score_cer
from src.utils.config import data_path, read_manifest

MAX_CER = 50.0
DURATION_BANDS = ((0, 5), (5, 10), (10, 20), (20, 61))
# Removed from a Devanagari word: chandrabindu, visarga, independent vowels, nukta,
# avagraha, vowel signs, halant, stress marks and the extra vowel letters.
DROP = re.compile("[ऀँः-औ़-्॑-ॗॠ-ॣॲ-ॷ]")
# Folded together: anusvara with na, aspirated with plain, dental with retroflex.
FOLD = str.maketrans({"ं": "न", "ङ": "न", "ञ": "न", "ण": "न", "द": "ड", "त": "ट", "थ": "ट",
                      "ठ": "ट", "ढ": "ड", "ध": "ड", "ख": "क", "घ": "ग", "छ": "च", "झ": "ज",
                      "भ": "ब", "श": "स", "ष": "स"})
LATIN = re.compile("[a-z]")
EXAMPLES = 12


def devanagari_skeleton(word: str) -> str:
    return DROP.sub("", unicodedata.normalize("NFD", word)).translate(FOLD)


class Skeleton:
    """Consonant skeleton of a transcript, in Devanagari whatever script it is written in."""

    def __init__(self, cmudict: dict[str, list[list[str]]]):
        self.cmudict = cmudict
        self.cache: dict[str, str] = {}

    def word(self, word: str) -> str:
        if word not in self.cache:
            if LATIN.search(word):
                found = pronunciations(word, self.cmudict)
                # Spelling variants differ mostly in vowels and halant, which the
                # skeleton drops; min() makes the choice among the rest repeatable.
                self.cache[word] = min(map(devanagari_skeleton, spellings(found[0]))) if found else word
            else:
                self.cache[word] = devanagari_skeleton(word)
        return self.cache[word]

    def __call__(self, text: str) -> str:
        return " ".join(form for form in map(self.word, text.split()) if form)


def character_error_rate(reference: str, hypothesis: str) -> float:
    return score_cer([(reference, hypothesis)], clean=False).wer


def error_rates(reference: str, hypothesis: str, skeleton: Skeleton) -> tuple[float, float]:
    """CER as written and in one script (as written when the skeleton is empty)."""
    as_written = character_error_rate(reference, hypothesis)
    reference_skeleton = skeleton(reference)
    if not reference_skeleton:
        return as_written, as_written
    return as_written, character_error_rate(reference_skeleton, skeleton(hypothesis))


def misaligned(reference: str, hypothesis: str, skeleton: Skeleton, max_cer: float = MAX_CER) -> bool:
    return min(error_rates(reference, hypothesis, skeleton)) > max_cer


def percentiles(values: list[float]) -> dict[str, float]:
    points = (50, 75, 90, 95, 97, 99)
    return {f"p{p}": round(float(v), 2) for p, v in zip(points, np.percentile(values, points))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default="data/processed/train_scores/baseline_rtx3050_best.jsonl")
    parser.add_argument("--manifest-dir", required=True, help="manifests to filter, e.g. the best iteration's")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cmudict", default="data/external/cmudict/cmudict.dict")
    parser.add_argument("--max-cer", type=float, default=MAX_CER)
    parser.add_argument("--report", default="results/train_filter_report.json")
    args = parser.parse_args()
    source, output = data_path(args.manifest_dir), data_path(args.output_dir)

    train = read_manifest(source / "train.jsonl")
    if any(row.get("split", "train") != "train" for row in train):
        raise ValueError("only training rows can be filtered")
    with data_path(args.scores).open(encoding="utf-8") as handle:
        scores = {row["id"]: row for row in map(json.loads, handle)}
    train_ids = {row["id"] for row in train}
    if not set(scores) <= train_ids:
        raise ValueError(f"{len(set(scores) - train_ids)} scored utterances are not in this training split")
    # The training loader skips the few utterances whose labels outrun their frames,
    # so they were never scored; they are kept here and skipped by the loader again.
    unscored = sorted(train_ids - set(scores))
    if len(unscored) > 0.001 * len(train):
        raise ValueError(f"{len(unscored)} training utterances have no score")

    skeleton = Skeleton(load_cmudict(data_path(args.cmudict)))
    rows = []
    for row in train:
        if row["id"] not in scores:
            continue
        reference = row.get("text_unnormalised", row["text"])
        hypothesis = scores[row["id"]]["hypothesis"]
        as_written, one_script = error_rates(reference, hypothesis, skeleton)
        rows.append({"id": row["id"], "duration": row["duration"], "reference": reference,
                     "hypothesis": hypothesis, "cer": round(as_written, 2),
                     "cer_one_script": round(one_script, 2), "drop": min(as_written, one_script) > args.max_cer})

    output.mkdir(parents=True, exist_ok=True)
    dropped = {row["id"] for row in rows if row["drop"]}
    with (output / "train.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in train:
            if row["id"] not in dropped:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for split in ("val", "test", "blind"):
        shutil.copy2(source / f"{split}.jsonl", output / f"{split}.jsonl")

    total = sum(row["duration"] for row in train)
    removed = sum(row["duration"] for row in rows if row["drop"])
    filter_cer = [min(row["cer"], row["cer_one_script"]) for row in rows]
    sample = random.Random(0)
    above = [r for r, c in zip(rows, filter_cer) if args.max_cer < c <= args.max_cer + 10]
    below = [r for r, c in zip(rows, filter_cer) if args.max_cer - 10 < c <= args.max_cer]
    fields = ("id", "duration", "cer", "cer_one_script", "reference", "hypothesis")
    report = {
        "rule": f"drop a training utterance when its CER under {data_path(args.scores).stem} is above "
                f"{args.max_cer} both as written and in one script (consonant skeletons)",
        "scores": data_path(args.scores).name,
        "manifests": args.manifest_dir,
        "output": args.output_dir,
        "max_cer": args.max_cer,
        "train_utterances": len(train),
        "scored_utterances": len(rows),
        "train_hours": round(total / 3600, 2),
        "dropped_utterances": len(dropped),
        "dropped_percent": round(100 * len(dropped) / len(train), 2),
        "dropped_hours": round(removed / 3600, 2),
        "dropped_audio_percent": round(100 * removed / total, 2),
        "kept_utterances": len(train) - len(dropped),
        "unscored_kept": unscored,
        "dropped_percent_by_duration": {
            f"{low}-{high} s": round(100 * sum(r["drop"] for r in band) / max(len(band), 1), 2)
            for low, high in DURATION_BANDS
            for band in [[r for r in rows if low <= r["duration"] < high]]},
        "percentiles": {"cer": percentiles([row["cer"] for row in rows]),
                        "cer_one_script": percentiles([row["cer_one_script"] for row in rows]),
                        "filter_cer": percentiles(filter_cer)},
        "examples_just_above": [{k: r[k] for k in fields} for r in sample.sample(above, min(EXAMPLES, len(above)))],
        "examples_just_below": [{k: r[k] for k in fields} for r in sample.sample(below, min(EXAMPLES, len(below)))],
        "dropped_ids": sorted(dropped),
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("train_utterances", "dropped_utterances", "dropped_percent",
                                                    "dropped_hours", "dropped_audio_percent")}))


if __name__ == "__main__":
    main()
