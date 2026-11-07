"""Build a KenLM n-gram model from the training manifest only.

Two builders produce the same kind of ARPA file. `lmplz` is used when it is on
PATH. It needs Boost, which is unavailable on some machines, so the built-in
builder implements the same method: interpolated Kneser-Ney smoothing with one
discount per order. `build_binary` then converts the ARPA for fast loading.

The manifest is deliberately fixed to ``train.jsonl``; validation, test and
blind transcripts are never read by this command.

    python -m src.language_model --config configs/data.yaml --order 5
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from src import ROOT
from src.utils.config import data_path, load_config

START, END, UNKNOWN = "<s>", "</s>", "<unk>"
Gram = tuple[str, ...]


def training_text(manifest: Path) -> list[str]:
    if manifest.name != "train.jsonl":
        raise ValueError(f"language-model input must be train.jsonl, got {manifest.name}")
    lines = []
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            entry = json.loads(line)
            if entry.get("split") != "train":
                raise ValueError(f"{manifest}:{line_number}: non-training row in training manifest")
            text = " ".join(entry.get("text", "").split())
            if text:
                lines.append(text)
    if not lines:
        raise ValueError(f"no non-empty training transcripts in {manifest}")
    return lines


def count_ngrams(lines: Iterable[str], order: int) -> list[Counter[Gram]]:
    counts: list[Counter[Gram]] = [Counter() for _ in range(order)]
    for line in lines:
        words = (START, *line.split(), END)
        for size in range(1, order + 1):
            table = counts[size - 1]
            for position in range(len(words) - size + 1):
                table[words[position : position + size]] += 1
    return counts


def kneser_ney_counts(counts: list[Counter[Gram]]) -> list[dict[Gram, int]]:
    """Replace lower-order counts by their number of distinct left contexts.

    The highest order keeps raw counts, and so do grams starting with <s>,
    because nothing can precede them.
    """
    adjusted: list[dict[Gram, int]] = []
    for level, table in enumerate(counts):
        if level == len(counts) - 1:
            adjusted.append(dict(table))
            continue
        contexts: Counter[Gram] = Counter()
        for gram in counts[level + 1]:
            contexts[gram[1:]] += 1
        adjusted.append({g: (c if g[0] == START else contexts[g]) for g, c in table.items()})
    return adjusted


def discount_for(table: dict[Gram, int]) -> float:
    """Ney's estimate D = n1 / (n1 + 2 n2), kept away from 0 and 1."""
    singletons = sum(1 for c in table.values() if c == 1)
    doubletons = sum(1 for c in table.values() if c == 2)
    if not singletons or not doubletons:
        return 0.5
    return min(max(singletons / (singletons + 2 * doubletons), 0.1), 0.9)


def smoothed_model(
    lines: Iterable[str], order: int
) -> tuple[list[dict[Gram, float]], dict[Gram, float], list[float]]:
    """Return log-free probabilities per order, backoffs and discounts.

    p(w|h) = max(a(hw) - D, 0) / A(h) + b(h) p(w|h'), where h' drops the oldest
    word and b(h) = D * types(h) / A(h). Unigrams interpolate with a uniform
    distribution over the vocabulary, so <unk> gets the leftover mass.
    """
    counts = count_ngrams(lines, order)
    adjusted = kneser_ney_counts(counts)
    adjusted[0] = {g: c for g, c in adjusted[0].items() if g != (START,)}
    discounts = [discount_for(table) for table in adjusted]

    total = sum(adjusted[0].values())
    vocabulary_size = len(adjusted[0]) + 1
    leftover = discounts[0] * len(adjusted[0]) / total
    uniform = leftover / vocabulary_size
    probabilities: list[dict[Gram, float]] = [
        {g: (c - discounts[0]) / total + uniform for g, c in adjusted[0].items()}
    ]
    probabilities[0][(UNKNOWN,)] = uniform

    backoffs: dict[Gram, float] = {}
    for level in range(1, order):
        table = adjusted[level]
        mass: dict[Gram, int] = defaultdict(int)
        types: dict[Gram, int] = defaultdict(int)
        for gram, value in table.items():
            mass[gram[:-1]] += value
            types[gram[:-1]] += 1
        discount = discounts[level]
        for history in mass:
            backoffs[history] = discount * types[history] / mass[history]
        lower = probabilities[level - 1]
        probabilities.append({
            gram: max(value - discount, 0) / mass[gram[:-1]] + backoffs[gram[:-1]] * lower[gram[1:]]
            for gram, value in table.items()
        })
    return probabilities, backoffs, discounts


def write_arpa(lines: Iterable[str], order: int, path: Path) -> dict:
    """Write an interpolated Kneser-Ney ARPA file and return its statistics."""
    probabilities, backoffs, discounts = smoothed_model(lines, order)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        sizes = [len(table) for table in probabilities]
        sizes[0] += 1  # <s> appears only as a context, so it is written separately
        out.write("\\data\\\n")
        for size, count in enumerate(sizes, 1):
            out.write(f"ngram {size}={count}\n")
        for level, table in enumerate(probabilities, 1):
            out.write(f"\n\\{level}-grams:\n")
            if level == 1:
                back = math.log10(backoffs[(START,)]) if order > 1 else 0.0
                out.write(f"-99\t{START}\t{back:.6f}\n")
            for gram, probability in table.items():
                text = f"{math.log10(probability):.6f}\t{' '.join(gram)}"
                if level < order:
                    back = backoffs.get(gram)
                    text += f"\t{math.log10(back) if back else 0.0:.6f}"
                out.write(text + "\n")
        out.write("\n\\end\\\n")
    return {
        "builder": "built-in interpolated Kneser-Ney",
        "ngrams_per_order": sizes,
        "discounts": [round(value, 4) for value in discounts],
    }


def build_model(
    manifest: Path,
    output_dir: Path,
    order: int = 5,
    lmplz: str = "lmplz",
    build_binary: str = "build_binary",
) -> tuple[Path, Path, dict]:
    if order < 1:
        raise ValueError("n-gram order must be positive")
    # The tool is often installed beside the interpreter, which may not be on PATH.
    binary_tool = shutil.which(build_binary) or shutil.which(build_binary, path=Path(sys.executable).parent)
    if binary_tool is None:
        raise FileNotFoundError(f"KenLM executable not found on PATH: {build_binary}")

    lines = training_text(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = output_dir / "train.txt"
    arpa = output_dir / f"train_{order}gram.arpa"
    binary = output_dir / f"train_{order}gram.klm"
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    if shutil.which(lmplz):
        subprocess.run(
            [lmplz, "-o", str(order), "--discount_fallback", "--text", str(corpus), "--arpa", str(arpa)],
            check=True,
        )
        report = {"builder": "lmplz"}
    else:
        report = write_arpa(lines, order, arpa)
    subprocess.run([binary_tool, str(arpa), str(binary)], check=True)

    words = [word for line in lines for word in line.split()]
    report.update({
        "source": "train.jsonl only",
        "normalisation": "the training transcript cleaner; both scripts preserved, no transliteration",
        "tokenisation": "whitespace-separated words",
        "order": order,
        "pruning": "none; with about 600 k words, every observed n-gram is kept",
        "sentences": len(lines),
        "tokens": len(words),
        "vocabulary": len(set(words)),
        "arpa_megabytes": round(arpa.stat().st_size / 1e6, 2),
        "binary_megabytes": round(binary.stat().st_size / 1e6, 2),
    })
    (output_dir / "unigrams.txt").write_text("\n".join(sorted(set(words))) + "\n", encoding="utf-8", newline="\n")
    return arpa, binary, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("data/language_model"))
    parser.add_argument("--lmplz", default="lmplz")
    parser.add_argument("--build-binary", default="build_binary")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = data_path(config["data"]["manifest_dir"]) / "train.jsonl"
    output_dir = data_path(args.output_dir)
    arpa, binary, report = build_model(manifest, output_dir, args.order, args.lmplz, args.build_binary)
    report.update({"arpa": arpa.name, "binary": binary.name})
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "lm_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
