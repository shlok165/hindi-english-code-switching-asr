"""Put every iteration's results side by side, with a fair val metric for choosing.

    python -m src.analysis.compare_iterations iteration1=results/evaluations \
        iteration2=results/iteration2/evaluations

Iterations that change the training targets cannot be compared on plain val WER:
one trained on script-normalised text is scored against normalised references,
the others against the original ones. So choices between iterations use
`val_wer_folded`: val WER after both sides are mapped through the train-only
script map, which makes a word count as correct in either script. The map comes
from training data only (src.data.script_normalisation), so this stays a val-only
decision. Test and blind numbers are reported, never used to choose.

Writes results/iterations.json and results/iterations.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import ROOT
from src.metrics.wer import score
from src.utils.config import data_path

SCRIPT_MAP = "data/processed/script_map/hi2en_train.json"
SPLITS = ("val", "test", "blind")
DECODERS = ("greedy", "beam")


def folded(text: str, mapping: dict[str, str]) -> str:
    return " ".join(mapping.get(word, word) for word in text.split())


def val_wer_folded(path: Path, mapping: dict[str, str]) -> float | None:
    if not path.exists():
        return None
    pairs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["reference"]:
                pairs.append((folded(row["reference"], mapping), folded(row["hypothesis"], mapping)))
    return round(score(pairs).wer, 2)


def collect(directory: Path, mapping: dict[str, str]) -> dict:
    found: dict = {}
    for split in SPLITS:
        for decoder in DECODERS:
            path = directory / f"{split}_{decoder}_metrics.json"
            if not path.exists():
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))
            row = {key: round(metrics[key], 2) for key in ("wer", "cer", "twer", "twer_approximate") if key in metrics}
            if split == "val":
                row["wer_folded"] = val_wer_folded(directory / f"val_{decoder}.jsonl", mapping)
            found[f"{split}_{decoder}"] = row
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("iterations", nargs="+", help="name=directory of evaluation results")
    args = parser.parse_args()
    mapping = json.loads(data_path(SCRIPT_MAP).read_text(encoding="utf-8"))

    table = {}
    for item in args.iterations:
        name, _, directory = item.partition("=")
        table[name] = collect(ROOT / directory, mapping)
    (ROOT / "results" / "iterations.json").write_text(json.dumps(table, indent=2), encoding="utf-8")

    lines = ["| Iteration | Val WER | Val WER, script-folded | Test WER | Test T-WER (approx.) "
             "| Blind WER | Blind T-WER | Blind CER |", "|---|---|---|---|---|---|---|---|"]
    for name, rows in table.items():
        val, test, blind = rows.get("val_beam", {}), rows.get("test_beam", {}), rows.get("blind_beam", {})
        cells = [val.get("wer"), val.get("wer_folded"), test.get("wer"), test.get("twer_approximate"),
                 blind.get("wer"), blind.get("twer"), blind.get("cer")]
        lines.append(f"| {name} | " + " | ".join("–" if c is None else f"{c:.2f}" for c in cells) + " |")
    text = "All numbers use KenLM beam search with each iteration's own val-tuned settings.\n\n" + "\n".join(lines) + "\n"
    (ROOT / "results" / "iterations.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
