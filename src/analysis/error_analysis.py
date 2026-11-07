"""Summarise word-level errors from an evaluation JSONL artifact.

Script labels and switch-context labels are Unicode-derived diagnostics, not
ground-truth language annotations. Transliteration and homophone causes require
manual review of the exported examples.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.metrics.wer import DEL, INS, MATCH, SUB, align

LATIN = re.compile("[A-Za-z]")
DEVANAGARI = re.compile("[ऀ-ॿ]")
OP_NAMES = {SUB: "substitution", DEL: "deletion", INS: "insertion"}


def script_of(word: str) -> str:
    has_latin = bool(LATIN.search(word))
    has_devanagari = bool(DEVANAGARI.search(word))
    if has_latin and has_devanagari:
        return "mixed"
    if has_latin:
        return "english_script"
    if has_devanagari:
        return "hindi_script"
    return "other"


def switch_context(words: list[str], index: int | None) -> str | None:
    if index is None:
        return None
    scripts = [script_of(word) for word in words]
    if index > 0 and scripts[index - 1] != scripts[index]:
        return f"{scripts[index - 1]}_to_{scripts[index]}"
    if index + 1 < len(words) and scripts[index] != scripts[index + 1]:
        return f"{scripts[index]}_to_{scripts[index + 1]}"
    return None


def analyse(path: Path, examples_per_category: int = 10, long_seconds: float = 15.0) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    utterances = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            utterances += 1
            reference = row["reference"].split()
            hypothesis = row["hypothesis"].split()
            duration = float(row.get("duration", 0.0))
            for operation, ref_index, hyp_index in align(reference, hypothesis):
                if operation == MATCH:
                    continue
                ref_word = reference[ref_index] if ref_index is not None else None
                hyp_word = hypothesis[hyp_index] if hyp_index is not None else None
                script = script_of(ref_word if ref_word is not None else hyp_word or "")
                long_label = "long" if duration >= long_seconds else "not_long"
                name = OP_NAMES[operation]
                switch = switch_context(reference, ref_index)
                totals[name] += 1
                totals[f"{name}:{script}"] += 1
                totals[f"{name}:{long_label}"] += 1
                if switch:
                    totals[f"{name}:switch_context:{switch}"] += 1
                category = f"{name}:{script}"
                if len(examples[category]) < examples_per_category:
                    examples[category].append(
                        {
                            "id": row["id"],
                            "reference": row["reference"],
                            "hypothesis": row["hypothesis"],
                            "reference_word": ref_word,
                            "hypothesis_word": hyp_word,
                            "duration": duration,
                            "switch_context": switch,
                        }
                    )
    return {
        "utterances": utterances,
        "long_seconds_threshold": long_seconds,
        "error_counts": dict(sorted(totals.items())),
        "examples": dict(sorted(examples.items())),
        "limitations": [
            "Script labels are derived from Unicode ranges, not language annotations.",
            "Switch context is a neighboring-script heuristic, not a switch-point gold label.",
            "Transliteration and homophone confusion require manual review.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--examples-per-category", type=int, default=10)
    parser.add_argument("--long-seconds", type=float, default=15.0)
    args = parser.parse_args()
    if args.examples_per_category < 0 or args.long_seconds <= 0:
        parser.error("example count must be non-negative and long-seconds must be positive")
    report = analyse(args.predictions, args.examples_per_category, args.long_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"utterances": report["utterances"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
