"""Character-level CTC vocabulary built from the training transcripts only.

Design decisions are explained in docs/stage_1a_report.md. In short:
- Blank is `<pad>`, id 0. CTC needs one blank, and Transformers takes it from
  `pad_token_id`.
- `|` is the word delimiter, so a space is an ordinary CTC symbol.
- `<unk>` exists so that the tokenizer is total, but rare characters are removed
  from the labels rather than mapped to it (Chadha et al.: frequency < 10), so
  the model is never trained to emit `<unk>`.
- `<s>` and `</s>` stay for Transformers compatibility and are never targets.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from transformers import Wav2Vec2CTCTokenizer

BLANK, BOS, EOS, UNK, DELIMITER = "<pad>", "<s>", "</s>", "<unk>", "|"
SPECIALS = [BLANK, BOS, EOS, UNK, DELIMITER]


def build_vocab(texts: Iterable[str], min_count: int = 10) -> tuple[dict[str, int], dict[str, int]]:
    """Return token -> id, plus the characters dropped as too rare."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(text.replace(" ", ""))
    if DELIMITER in counts:
        raise ValueError("'|' is reserved for the word delimiter; clean transcripts first")
    kept = sorted(char for char, n in counts.items() if n >= min_count)
    dropped = {char: n for char, n in counts.items() if n < min_count}
    return {token: i for i, token in enumerate(SPECIALS + kept)}, dropped


def make_tokenizer(vocab_file: Path) -> Wav2Vec2CTCTokenizer:
    return Wav2Vec2CTCTokenizer(
        str(vocab_file),
        bos_token=BOS,
        eos_token=EOS,
        unk_token=UNK,
        pad_token=BLANK,
        word_delimiter_token=DELIMITER,
        do_lower_case=False,
    )


def save_tokenizer(vocab: dict[str, int], directory: Path) -> Wav2Vec2CTCTokenizer:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "vocab.json"
    path.write_text(json.dumps(vocab, ensure_ascii=False, indent=0), encoding="utf-8")
    return make_tokenizer(path)


def load_tokenizer(directory: str | Path) -> Wav2Vec2CTCTokenizer:
    return make_tokenizer(Path(directory) / "vocab.json")


def encode(text: str, vocab: dict[str, int]) -> list[int]:
    """Text to label ids. Characters outside the vocabulary are removed."""
    collapsed = " ".join("".join(c for c in text if c == " " or c in vocab).split())
    return [vocab[c] for c in collapsed.replace(" ", DELIMITER)]


def decode_labels(ids: Iterable[int], vocab: dict[str, int]) -> str:
    """Label ids to text, with no CTC collapsing: repeated letters are real."""
    inverse = {index: token for token, index in vocab.items()}
    return "".join(inverse[i] for i in ids).replace(DELIMITER, " ")


def ctc_collapse(frame_ids: Iterable[int], blank: int) -> list[int]:
    """Merge repeated frame ids, then remove blanks, in that order."""
    collapsed, previous = [], None
    for index in frame_ids:
        if index != previous and index != blank:
            collapsed.append(index)
        previous = index
    return collapsed


def main() -> None:
    """Build the vocabulary from train.jsonl alone and verify a full round trip."""
    import argparse

    from src import ROOT
    from src.utils.config import data_path, load_config

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    data = load_config(args.config)["data"]

    manifest = data_path(data["manifest_dir"]) / "train.jsonl"
    with manifest.open(encoding="utf-8") as handle:
        texts = [t for t in (json.loads(line)["text"] for line in handle) if t]
    vocab, dropped = build_vocab(texts, data["vocab_min_count"])
    directory = data_path(data["tokenizer_dir"])
    tokenizer = save_tokenizer(vocab, directory)

    changed = [t for t in texts if decode_labels(encode(t, vocab), vocab) != t]
    sample = texts[:: max(len(texts) // 500, 1)]
    agrees = all(tokenizer(t).input_ids == encode(t, vocab) for t in sample
                 if all(c == " " or c in vocab for c in t))
    groups = {"devanagari": 0, "latin": 0, "digit": 0, "symbol": 0}
    for token in vocab:
        if token in SPECIALS:
            continue
        if "ऀ" <= token <= "ॿ":
            groups["devanagari"] += 1
        elif token.isascii() and token.isalpha():
            groups["latin"] += 1
        elif token.isdigit():
            groups["digit"] += 1
        else:
            groups["symbol"] += 1
    report = {
        "source": "train.jsonl only",
        "transcripts": len(texts),
        "vocab_size": len(vocab),
        "groups": groups,
        "symbols": [t for t in vocab if t not in SPECIALS and not t.isalnum()
                    and not "ऀ" <= t <= "ॿ"],
        "ids": {"blank": vocab[BLANK], "unk": vocab[UNK], "delimiter": vocab[DELIMITER]},
        "dropped_rare_characters": dropped,
        "transcripts_changed_by_dropping": len(changed),
        "round_trip_exact_percent": round(100 * (1 - len(changed) / len(texts)), 3),
        "hf_tokenizer_agrees_on_sample": agrees,
    }
    output = ROOT / "docs" / "vocab_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
