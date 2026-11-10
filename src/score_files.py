"""Score reference and hypothesis files without a model, audio or dataset.

    python -m src.score_files --ref ref.txt --hyp hyp.txt --output-dir results/cpu_eval/scoring/my_run
    python -m src.score_files --predictions results/iteration4/evaluations/val_beam.jsonl \
        --output-dir results/cpu_eval/scoring/iteration4_val_beam
    python -m src.score_files --pairs pairs.csv --split blind --output-dir ...

Inputs:
- `--ref` and `--hyp`: Kaldi-style files, `utterance-id text` per line, matched by
  id (`--no-ids`: plain text, paired line by line).
- `--pairs`: CSV with `reference` and `hypothesis` columns, optionally `id` and `duration`.
- `--predictions`: JSON lines from `src.evaluate` or `src.inference --manifest`
  (fields `reference`, `hypothesis` or `hypothesis_raw`, optionally `id`,
  `duration`, `split`, `reference_cleaned`).

Three normalisations are scored separately and named as such in the output:
- `official`: the organisers' compute_wer.py rules (src.metrics.official):
  lower-case both sides, apply punct.mapping.txt to the hypothesis, skip empty
  references. `wer` and `cer` use this. T-WER additionally maps the hypothesis
  through hin2engwords.txt. It is official only for blind, whose references
  were annotated for it; for test it is an approximation that maps both sides
  (`twer_approximate`); for val or custom data it is not reported unless asked.
- `cleaned`: the training cleaner (src.data.preprocessing.clean_transcript) on
  both sides, giving `wer_cleaned` and `cer_cleaned`.
- `script_folded`: both sides mapped through the train-only script map
  (data/processed/script_map/hi2en_train.json) so a word counts as correct in
  either script. Reported only when that map file exists.

Outputs, in --output-dir (existing files are never overwritten unless --force):
summary.json, utterances.csv (per-utterance counts and both scored strings),
word_errors.csv (every substitution, deletion and insertion with the script of
each word) and confusions.csv (error pairs by frequency).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.data.preprocessing import clean_transcript
from src.metrics.cer import count_characters
from src.metrics.official import (
    INFERENCE_DIR,
    ScoringMaps,
    apply_map,
    load_maps,
    normalise_hypothesis,
    normalise_reference,
)
from src.metrics.wer import DEL, INS, MATCH, SUB, Counts, align
from src.utils.config import data_path

SCRIPT_MAP = "data/processed/script_map/hi2en_train.json"
TWER_MODES = ("auto", "none", "official", "approximate")
OUTPUT_FILES = ("summary.json", "utterances.csv", "word_errors.csv", "confusions.csv")
OP_NAMES = {SUB: "substitution", DEL: "deletion", INS: "insertion"}
NORMALISATIONS = {
    "official": "lower-case both sides; punct.mapping.txt on the hypothesis; empty references skipped "
                "(the organisers' compute_wer.py rules, src.metrics.official)",
    "twer": "official normalisation, then hin2engwords.txt on the hypothesis (blind) or on both sides "
            "(test, approximate)",
    "cleaned": "src.data.preprocessing.clean_transcript on both sides (training cleaning)",
    "script_folded": "official normalisation, then both sides mapped through the train-only script map "
                     "data/processed/script_map/hi2en_train.json",
}


class ScoringError(Exception):
    """Bad or mismatched input files, reported without a traceback."""


@dataclass
class Utterance:
    id: str
    reference: str
    hypothesis: str
    duration: float | None = None
    reference_cleaned: str | None = None


# ----------------------------------------------------------------------------- reading


def read_kaldi(path: Path) -> dict[str, str]:
    """`id text` per line. Returns an ordered mapping; a line with only an id is empty text."""
    rows: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if parts[0] in rows:
            raise ScoringError(f"{path}:{number}: duplicate utterance id {parts[0]!r}")
        rows[parts[0]] = parts[1] if len(parts) > 1 else ""
    if not rows:
        raise ScoringError(f"{path}: no utterances")
    return rows


def read_ref_hyp(ref_path: Path, hyp_path: Path, with_ids: bool = True) -> list[Utterance]:
    for path in (ref_path, hyp_path):
        if not path.is_file():
            raise ScoringError(f"file not found: {path}")
    if not with_ids:
        refs = ref_path.read_text(encoding="utf-8").splitlines()
        hyps = hyp_path.read_text(encoding="utf-8").splitlines()
        if len(refs) != len(hyps):
            raise ScoringError(f"{ref_path} has {len(refs)} lines but {hyp_path} has {len(hyps)}")
        return [Utterance(str(i), r, h) for i, (r, h) in enumerate(zip(refs, hyps), 1)]
    refs, hyps = read_kaldi(ref_path), read_kaldi(hyp_path)
    missing = [i for i in refs if i not in hyps]
    if missing:
        raise ScoringError(f"{hyp_path} has no hypothesis for {len(missing)} reference ids, "
                           f"first {missing[:3]}")
    extra = [i for i in hyps if i not in refs]
    if extra:
        raise ScoringError(f"{hyp_path} has {len(extra)} ids that are not in {ref_path}, first {extra[:3]}")
    return [Utterance(i, text, hyps[i]) for i, text in refs.items()]


def read_pairs_csv(path: Path) -> list[Utterance]:
    if not path.is_file():
        raise ScoringError(f"file not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        for column in ("reference", "hypothesis"):
            if column not in columns:
                raise ScoringError(f"{path}: needs a {column!r} column; found {columns}")
        rows = list(reader)
    if not rows:
        raise ScoringError(f"{path}: no utterances")
    return [Utterance(row.get("id") or str(n), row["reference"] or "", row["hypothesis"] or "",
                      float(row["duration"]) if row.get("duration") else None)
            for n, row in enumerate(rows, 1)]


def read_predictions(path: Path) -> tuple[list[Utterance], dict[str, Any]]:
    """JSONL from src.evaluate or src.inference --manifest. Returns records and what was found."""
    if not path.is_file():
        raise ScoringError(f"file not found: {path}")
    records: list[Utterance] = []
    info: dict[str, Any] = {"split": None, "hypothesis_field": None, "failed_rows": 0}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ScoringError(f"{path}:{number}: not JSON ({error})") from None
            if "error" in row and row.get("hypothesis") is None and row.get("hypothesis_raw") is None:
                info["failed_rows"] += 1  # a transcription that failed (src.inference --manifest); nothing to score
                continue
            if row.get("reference") is None:
                raise ScoringError(f"{path}:{number}: no 'reference' field; the manifest given to "
                                   "src.inference needs a reference column for scoring")
            field = "hypothesis_raw" if "hypothesis_raw" in row else "hypothesis"
            if field not in row:
                raise ScoringError(f"{path}:{number}: no 'hypothesis' field")
            info["hypothesis_field"] = info["hypothesis_field"] or field
            info["split"] = info["split"] or row.get("split")
            duration = row.get("duration")
            records.append(Utterance(str(row.get("id", number)), row["reference"], row[field] or "",
                                     float(duration) if duration is not None else None,
                                     row.get("reference_cleaned")))
    if not records:
        raise ScoringError(f"{path}: no scorable rows")
    return records, info


# ----------------------------------------------------------------------------- scoring


def counts_of(reference: str, hypothesis: str) -> tuple[Counts, list[tuple[str, int | None, int | None]]]:
    ref, hyp = reference.split(), hypothesis.split()
    ops = align(ref, hyp)
    tags = [op for op, _, _ in ops]
    return Counts(len(ref), tags.count(MATCH), tags.count(SUB), tags.count(DEL), tags.count(INS)), ops


def counts_dict(counts: Counts) -> dict[str, Any]:
    return {"reference_words": counts.ref_words, "errors": counts.errors, "substitutions": counts.substitutions,
            "deletions": counts.deletions, "insertions": counts.insertions}


def resolve_twer_mode(mode: str, split: str | None) -> str:
    if mode not in TWER_MODES:
        raise ScoringError(f"unknown T-WER mode {mode!r}; choose one of {TWER_MODES}")
    if mode != "auto":
        return mode
    return {"blind": "official", "test": "approximate"}.get(split or "", "none")


def load_script_map(path: str | Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    path = data_path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def score_records(
    records: Sequence[Utterance],
    maps: ScoringMaps | None,
    script_map: dict[str, str] | None = None,
    twer: str = "none",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (summary, per-utterance rows, word-error rows)."""
    from src.analysis.error_analysis import script_of

    punctuation = maps.punctuation if maps else {}
    official = Counts()
    characters = Counts()
    cleaned_words = Counts()
    cleaned_chars = Counts()
    folded = Counts()
    twer_counts = Counts()
    scored = empty_hypotheses = 0
    utterances: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    confusions: Counter[tuple[str, str | None, str | None]] = Counter()

    for record in records:
        reference = normalise_reference(record.reference)
        if not reference:
            continue
        scored += 1
        hypothesis = normalise_hypothesis(record.hypothesis, punctuation)
        empty_hypotheses += not hypothesis
        counts, ops = counts_of(reference, hypothesis)
        chars = count_characters(reference, hypothesis)
        official, characters = official + counts, characters + chars

        clean_ref = record.reference_cleaned if record.reference_cleaned is not None else clean_transcript(record.reference)
        clean_hyp = clean_transcript(record.hypothesis)
        if clean_ref:
            cleaned_words = cleaned_words + counts_of(clean_ref, clean_hyp)[0]
            cleaned_chars = cleaned_chars + count_characters(clean_ref, clean_hyp)

        row: dict[str, Any] = {
            "id": record.id, "duration": record.duration, **counts_dict(counts),
            "wer": round(counts.wer, 2), "reference_characters": chars.ref_words,
            "character_errors": chars.errors, "cer": round(chars.wer, 2),
        }
        if twer != "none" and maps:
            mapped_hyp = apply_map(hypothesis, maps.words)
            mapped_ref = apply_map(reference, maps.words) if twer == "approximate" else reference
            twer_utt = counts_of(mapped_ref, mapped_hyp)[0]
            twer_counts = twer_counts + twer_utt
            row["twer_errors"] = twer_utt.errors
        if script_map is not None:
            folded_utt = counts_of(apply_map(reference, script_map), apply_map(hypothesis, script_map))[0]
            folded = folded + folded_utt
            row["folded_errors"] = folded_utt.errors
        row["reference"], row["hypothesis"] = reference, hypothesis
        utterances.append(row)

        ref_words, hyp_words = reference.split(), hypothesis.split()
        for op, ref_index, hyp_index in ops:
            if op == MATCH:
                continue
            ref_word = ref_words[ref_index] if ref_index is not None else None
            hyp_word = hyp_words[hyp_index] if hyp_index is not None else None
            confusions[(OP_NAMES[op], ref_word, hyp_word)] += 1
            errors.append({
                "id": record.id, "operation": OP_NAMES[op],
                "reference_position": ref_index, "hypothesis_position": hyp_index,
                "reference_word": ref_word, "hypothesis_word": hyp_word,
                "reference_script": script_of(ref_word) if ref_word else None,
                "hypothesis_script": script_of(hyp_word) if hyp_word else None,
            })

    summary: dict[str, Any] = {
        "utterances": len(records),
        "scored_utterances": scored,
        "empty_references": len(records) - scored,
        "empty_hypotheses": empty_hypotheses,
        "wer": official.wer,
        "cer": characters.wer,
        "wer_cleaned": cleaned_words.wer,
        "cer_cleaned": cleaned_chars.wer,
        "reference_words": official.ref_words,
        "counts": {
            "official": counts_dict(official),
            "official_characters": counts_dict(characters),
            "cleaned": counts_dict(cleaned_words),
            "cleaned_characters": counts_dict(cleaned_chars),
        },
        "normalisation": dict(NORMALISATIONS),
    }
    if twer == "official" and maps:
        summary["twer"] = twer_counts.wer
        summary["counts"]["twer"] = counts_dict(twer_counts)
    elif twer == "approximate" and maps:
        summary["twer_approximate"] = twer_counts.wer
        summary["counts"]["twer_approximate"] = counts_dict(twer_counts)
    else:
        summary["twer"] = None
        summary["twer_reason"] = ("the word maps were not found" if maps is None and twer != "none" else
                                  "not computed: official for blind, approximate for test, --twer to force")
    if maps is None:
        summary["normalisation"]["official"] += " [punct.mapping.txt not found: no punctuation mapping applied]"
    if script_map is not None:
        summary["wer_folded"] = folded.wer
        summary["counts"]["script_folded"] = counts_dict(folded)
    else:
        summary["wer_folded"] = None
        summary["wer_folded_reason"] = f"{SCRIPT_MAP} not found; build it with python -m src.data.script_normalisation"
    summary["confusions"] = [{"operation": op, "reference_word": r, "hypothesis_word": h, "count": n}
                             for (op, r, h), n in confusions.most_common()]
    return summary, utterances, errors


# ----------------------------------------------------------------------------- writing


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir: Path, summary: dict[str, Any], utterances: list[dict[str, Any]],
                  errors: list[dict[str, Any]], force: bool = False) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not force:
        raise ScoringError(f"refusing to overwrite {', '.join(p.name for p in existing)} in {output_dir}; "
                           "choose another --output-dir or pass --force")
    confusions = summary["confusions"]
    summary_file = {key: value for key, value in summary.items() if key != "confusions"}
    summary_file["top_confusions"] = confusions[:40]
    (output_dir / "summary.json").write_text(json.dumps(summary_file, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    columns = list(utterances[0].keys()) if utterances else ["id"]
    write_csv(output_dir / "utterances.csv", utterances, columns)
    write_csv(output_dir / "word_errors.csv", errors, [
        "id", "operation", "reference_position", "hypothesis_position", "reference_word", "hypothesis_word",
        "reference_script", "hypothesis_script"])
    write_csv(output_dir / "confusions.csv", confusions, ["operation", "reference_word", "hypothesis_word", "count"])
    return [output_dir / name for name in OUTPUT_FILES]


# ----------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ref", type=Path, help="reference file, `id text` per line (with --hyp)")
    source.add_argument("--pairs", type=Path, help="CSV with reference and hypothesis columns")
    source.add_argument("--predictions", type=Path, help="JSONL from src.evaluate or src.inference --manifest")
    parser.add_argument("--hyp", type=Path, help="hypothesis file in the same format as --ref")
    parser.add_argument("--no-ids", action="store_true", help="--ref/--hyp are plain text, paired by line")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="where summary.json and the CSV files go, e.g. results/cpu_eval/scoring/<name>")
    parser.add_argument("--split", choices=("val", "test", "blind", "custom"),
                        help="decides T-WER in --twer auto; read from --predictions when absent")
    parser.add_argument("--twer", choices=TWER_MODES, default="auto",
                        help="official (hypothesis mapped; blind), approximate (both sides; test) or none")
    parser.add_argument("--maps-dir", default=INFERENCE_DIR,
                        help="directory with punct.mapping.txt and hin2engwords.txt (default: %(default)s)")
    parser.add_argument("--script-map", default=SCRIPT_MAP,
                        help="train-only script map for the folded WER; skipped when absent")
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.ref and not args.hyp:
        parser.error("--ref needs --hyp")
    if args.hyp and not args.ref:
        parser.error("--hyp needs --ref")
    try:
        info: dict[str, Any] = {}
        if args.ref:
            records = read_ref_hyp(args.ref, args.hyp, with_ids=not args.no_ids)
            source = {"reference": str(args.ref), "hypothesis": str(args.hyp)}
        elif args.pairs:
            records = read_pairs_csv(args.pairs)
            source = {"pairs": str(args.pairs)}
        else:
            records, info = read_predictions(args.predictions)
            source = {"predictions": str(args.predictions), **info}
        split = args.split or info.get("split")
        twer = resolve_twer_mode(args.twer, split)
        maps_dir = data_path(args.maps_dir)
        maps = load_maps(maps_dir) if (maps_dir / "punct.mapping.txt").is_file() else None
        script_map = load_script_map(args.script_map)
        summary, utterances, errors = score_records(records, maps, script_map, twer)
        summary = {"source": source, "split": split, "twer_mode": twer,
                   "maps_dir": str(maps_dir) if maps else None, **summary}
        written = write_outputs(args.output_dir, summary, utterances, errors, args.force)
    except ScoringError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    shown = {key: summary[key] for key in ("utterances", "scored_utterances", "wer", "cer", "wer_cleaned",
                                            "cer_cleaned", "twer", "twer_approximate", "wer_folded") if key in summary}
    shown = {key: round(value, 4) if isinstance(value, float) else value for key, value in shown.items()}
    print(json.dumps(shown, ensure_ascii=False, indent=2))
    print("wrote " + ", ".join(str(path) for path in written), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
