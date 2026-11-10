"""Compare per-word log10 scores from src/arpa_lm.py with real KenLM's.

    python scripts/compare_lm_scores.py --arpa results/cpu_eval/lm_check/arpa_scores.jsonl \
        --kenlm results/cpu_eval/lm_check/kenlm_scores.jsonl \
        --output results/cpu_eval/lm_check/comparison.json

Writes the maximum and mean absolute difference and every word whose scores
differ by more than the tolerance. Exits 1 when any word does.
"""

import argparse
import json
import sys
from pathlib import Path

TOLERANCE = 1e-4  # KenLM stores probabilities as float32; ARPA has 6 decimals


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arpa", required=True, type=Path)
    parser.add_argument("--kenlm", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()
    ours, theirs = read(args.arpa), read(args.kenlm)
    if [row["sentence"] for row in ours] != [row["sentence"] for row in theirs]:
        sys.exit("the two files do not hold the same sentences in the same order")

    differences, mismatches = [], []
    for mine, reference in zip(ours, theirs):
        words = mine["sentence"].split() + ["</s>"]
        if len(mine["words"]) != len(reference["words"]):
            sys.exit(f"word counts differ for {mine['sentence']!r}")
        for word, a, b in zip(words, mine["words"], reference["words"]):
            difference = abs(a - b)
            differences.append(difference)
            if difference > args.tolerance:
                mismatches.append({"sentence": mine["sentence"], "word": word, "arpa": a, "kenlm": b})
    sentence_differences = [abs(a["log10"] - b["log10"]) for a, b in zip(ours, theirs)]
    report = {
        "sentences": len(ours),
        "words_scored": len(differences),
        "tolerance": args.tolerance,
        "max_word_difference": max(differences),
        "mean_word_difference": sum(differences) / len(differences),
        "max_sentence_difference": max(sentence_differences),
        "words_over_tolerance": len(mismatches),
        "mismatches": mismatches[:50],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "mismatches"}, indent=2))
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
