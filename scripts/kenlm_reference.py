"""Score sentences with real KenLM, as the reference for src/arpa_lm.py.

Needs only the `kenlm` package (no torch, no repository imports), so it can run
wherever kenlm installs, for example WSL Ubuntu when Windows has no wheel:

    python scripts/kenlm_reference.py --klm data/language_model/train_5gram.klm \
        --sentences results/cpu_eval/lm_check/arpa_scores.jsonl \
        --output results/cpu_eval/lm_check/kenlm_scores.jsonl

`--sentences` may be a JSONL file with a "sentence" field (such as the output of
`python -m src.arpa_lm`) or plain text, one sentence per line.
"""

import argparse
import json
from pathlib import Path

import kenlm


def read_sentences(path: Path) -> list[str]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if lines and lines[0].lstrip().startswith("{"):
        return [json.loads(line)["sentence"] for line in lines]
    return [line.strip() for line in lines]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--klm", required=True, type=Path)
    parser.add_argument("--sentences", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model = kenlm.Model(str(args.klm))
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for sentence in read_sentences(args.sentences):
            words = [score for score, _, _ in model.full_scores(sentence, bos=True, eos=True)]
            row = {"sentence": sentence, "log10": model.score(sentence, bos=True, eos=True), "words": words}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"kenlm {kenlm.__file__}: order {model.order}, scored into {args.output}")


if __name__ == "__main__":
    main()
