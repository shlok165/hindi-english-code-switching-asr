"""Categorise word-level errors from an evaluation JSONL artifact.

    python -m src.analysis.error_analysis --predictions results/evaluations/blind_beam.jsonl
        --output results/error_analysis/blind_beam.json

Every word-level label here is derived, not gold, because MUCS has no language
tags. A word is English if it contains Latin letters or is a key of the
organisers' word map, Hindi if it is otherwise Devanagari. Transliteration and
homophone labels use the phonetic rules below, so treat them as estimates.

Substitutions on an English reference word are split as follows:
- `script_only`: the hypothesis is the word's listed transliteration, which is
  correct speech recognition that only T-WER forgives.
- `incorrect_transliteration`: a Devanagari hypothesis that sounds close to the
  English word but is not its listed spelling.
Homophone confusion is a Devanagari substitution with an identical phonetic key.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from rapidfuzz.fuzz import ratio

from src.metrics.wer import DEL, INS, MATCH, SUB, align

LATIN = re.compile("[A-Za-z]")
DEVANAGARI = re.compile("[ऀ-ॿ]")
OP_NAMES = {SUB: "substitution", DEL: "deletion", INS: "insertion"}
DURATION_BUCKETS = ((0, 5, "under 5 s"), (5, 10, "5-10 s"), (10, 20, "10-20 s"), (20, 1e9, "over 20 s"))
TRANSLITERATION_SIMILARITY = 70.0  # Skeleton similarity; a judgement call

# Consonant sounds. Aspiration and dental/retroflex differences collapse because
# Hindi spellings of English words vary across exactly those contrasts.
DEVANAGARI_SOUNDS = {
    "क": "k", "ख": "k", "ग": "g", "घ": "g", "च": "c", "छ": "c", "ज": "j", "झ": "j",
    "ट": "t", "ठ": "t", "ड": "d", "ढ": "d", "ण": "n", "त": "t", "थ": "t", "द": "d",
    "ध": "d", "न": "n", "प": "p", "फ": "f", "ब": "b", "भ": "b", "म": "m", "य": "y",
    "र": "r", "ल": "l", "व": "v", "श": "s", "ष": "s", "स": "s", "ह": "h",
}
NUKTA_SOUNDS = {"ज": "j", "फ": "f", "क": "k", "ख": "k", "ग": "g", "ड": "d", "ढ": "d"}
LATIN_RULES = (("ph", "f"), ("sh", "s"), ("ch", "c"), ("th", "t"), ("ck", "k"), ("qu", "kv"),
               ("x", "ks"), ("w", "v"), ("q", "k"), ("z", "j"))
HOMOPHONE_FOLDS = str.maketrans({"ी": "ि", "ू": "ु", "ई": "इ", "ऊ": "उ", "ँ": "ं", "़": None})


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


def language_of(word: str, word_map: Mapping[str, str]) -> str:
    """Derived language: `en`, `hi`, or `other` for digits and symbols."""
    script = script_of(word)
    if script in ("english_script", "mixed") or word in word_map:
        return "en"
    return "hi" if script == "hindi_script" else "other"


def consonant_skeleton(word: str) -> str:
    """Rough sound of a word in either script, keeping consonants only."""
    if LATIN.search(word):
        word = word.lower()
        for old, new in LATIN_RULES:
            word = word.replace(old, new)
        word = re.sub(r"c(?=[eiy])", "s", word).replace("c", "k")
        word = re.sub("[^a-z]", "", word)
        return re.sub("[aeiouy]", "", word)
    sounds = []
    for character in unicodedata.normalize("NFC", word):
        if character in DEVANAGARI_SOUNDS:
            sound = DEVANAGARI_SOUNDS[character]
            sounds.append("k" if sound == "c" else sound)
    return "".join(sounds).replace("y", "")


def homophone_key(word: str) -> str:
    """Fold spellings that a listener cannot hear apart in fluent Hindi."""
    decomposed = unicodedata.normalize("NFD", word).replace("यी", "ई").replace("ये", "ए")
    return decomposed.translate(HOMOPHONE_FOLDS)


def is_listed_transliteration(reference: str, hypothesis: str, word_map: Mapping[str, str]) -> bool:
    return word_map.get(hypothesis) == reference


def is_near_transliteration(reference: str, hypothesis: str) -> bool:
    if script_of(reference) != "english_script" or script_of(hypothesis) != "hindi_script":
        return False
    first, second = consonant_skeleton(reference), consonant_skeleton(hypothesis)
    return bool(first and second) and ratio(first, second) >= TRANSLITERATION_SIMILARITY


def is_homophone(reference: str, hypothesis: str) -> bool:
    if script_of(reference) != "hindi_script" or script_of(hypothesis) != "hindi_script":
        return False
    return homophone_key(reference) == homophone_key(hypothesis)


def substitution_type(reference: str, hypothesis: str, word_map: Mapping[str, str]) -> str:
    if is_listed_transliteration(reference, hypothesis, word_map):
        return "script_only"
    if is_near_transliteration(reference, hypothesis):
        return "incorrect_transliteration"
    if is_homophone(reference, hypothesis):
        return "homophone_confusion"
    return "other_substitution"


def switch_context(words: list[str], index: int | None, word_map: Mapping[str, str] | None = None) -> str | None:
    """Return the switch next to a word, as `hi_to_en` or `en_to_hi`, if any."""
    if index is None:
        return None
    tags = [language_of(word, word_map or {}) for word in words]
    if index > 0 and {tags[index - 1], tags[index]} == {"hi", "en"}:
        return f"{tags[index - 1]}_to_{tags[index]}"
    if index + 1 < len(words) and {tags[index], tags[index + 1]} == {"hi", "en"}:
        return f"{tags[index]}_to_{tags[index + 1]}"
    return None


def duration_bucket(seconds: float) -> str:
    return next(name for low, high, name in DURATION_BUCKETS if low <= seconds < high)


def rate(errors: int, total: int) -> float:
    return round(100 * errors / total, 2) if total else 0.0


def analyse(path: Path, word_map: Mapping[str, str] | None = None, examples_per_category: int = 10) -> dict[str, Any]:
    word_map = word_map or {}
    categories: Counter[str] = Counter()
    words_by_language: Counter[str] = Counter()
    errors_by_language: Counter[str] = Counter()
    words_by_switch: Counter[str] = Counter()
    errors_by_switch: Counter[str] = Counter()
    words_by_duration: Counter[str] = Counter()
    errors_by_duration: Counter[str] = Counter()
    confusions: Counter[tuple[str, str]] = Counter()
    deleted: Counter[str] = Counter()
    inserted: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    utterances = 0

    def remember(category: str, row: dict, reference: str | None, hypothesis: str | None) -> None:
        if len(examples[category]) < examples_per_category:
            examples[category].append({
                "id": row["id"], "duration": row.get("duration"),
                "reference": row["reference"], "hypothesis": row["hypothesis"],
                "reference_word": reference, "hypothesis_word": hypothesis,
            })

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            reference, hypothesis = row["reference"].split(), row["hypothesis"].split()
            if not reference:
                continue
            utterances += 1
            bucket = duration_bucket(float(row.get("duration", 0.0)))
            words_by_duration[bucket] += len(reference)
            for word_index in range(len(reference)):
                words_by_language[language_of(reference[word_index], word_map)] += 1
                switch = switch_context(reference, word_index, word_map)
                if switch:
                    words_by_switch[switch] += 1

            for operation, ref_index, hyp_index in align(reference, hypothesis):
                if operation == MATCH:
                    continue
                ref_word = reference[ref_index] if ref_index is not None else None
                hyp_word = hypothesis[hyp_index] if hyp_index is not None else None
                categories[OP_NAMES[operation]] += 1
                errors_by_duration[bucket] += 1
                if operation == INS:
                    inserted[hyp_word] += 1
                    remember("insertion", row, ref_word, hyp_word)
                    continue
                language = language_of(ref_word, word_map)
                errors_by_language[language] += 1
                switch = switch_context(reference, ref_index, word_map)
                if switch:
                    errors_by_switch[switch] += 1
                    remember(switch, row, ref_word, hyp_word)
                remember(f"{language}_word", row, ref_word, hyp_word)
                if operation == DEL:
                    deleted[ref_word] += 1
                    remember("deletion", row, ref_word, hyp_word)
                    continue
                kind = substitution_type(ref_word, hyp_word, word_map)
                categories[kind] += 1
                confusions[(ref_word, hyp_word)] += 1
                remember(kind, row, ref_word, hyp_word)

    total_errors = categories[OP_NAMES[SUB]] + categories[OP_NAMES[DEL]] + categories[OP_NAMES[INS]]
    total_words = sum(words_by_language.values())
    return {
        "utterances": utterances,
        "reference_words": total_words,
        "total_errors": total_errors,
        "wer": rate(total_errors, total_words),
        "error_types": dict(categories.most_common()),
        "by_language": {
            language: {"words": words_by_language[language], "errors": errors_by_language[language],
                       "error_rate": rate(errors_by_language[language], words_by_language[language]),
                       "share_of_word_errors": rate(errors_by_language[language], sum(errors_by_language.values()))}
            for language in ("hi", "en", "other")
        },
        "by_switch": {
            switch: {"words": words_by_switch[switch], "errors": errors_by_switch[switch],
                     "error_rate": rate(errors_by_switch[switch], words_by_switch[switch])}
            for switch in ("hi_to_en", "en_to_hi")
        },
        "by_duration": {
            name: {"words": words_by_duration[name], "errors": errors_by_duration[name],
                   "wer": rate(errors_by_duration[name], words_by_duration[name])}
            for _, _, name in DURATION_BUCKETS
        },
        "top_confusions": [[r, h, n] for (r, h), n in confusions.most_common(40)],
        "top_deletions": deleted.most_common(25),
        "top_insertions": inserted.most_common(25),
        "examples": dict(sorted(examples.items())),
        "limitations": [
            "Language labels are derived from script and the word map, not gold annotations.",
            "Switch rates count reference words next to a Hindi-English script boundary.",
            "Rates exclude insertions unless stated, because an insertion has no reference word.",
            "Transliteration and homophone labels are phonetic heuristics; sample them before relying on them.",
        ],
    }


def main() -> None:
    from src.metrics.official import load_maps

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--examples-per-category", type=int, default=10)
    args = parser.parse_args()
    if args.examples_per_category < 0:
        parser.error("example count must be non-negative")
    with args.predictions.open(encoding="utf-8") as handle:
        split = json.loads(handle.readline()).get("split")
    # The official word map was built from blind references, so it only labels
    # splits that are scored with it. A val analysis must not carry blind vocabulary.
    word_map = load_maps().words if split in ("test", "blind") else {}
    report = analyse(args.predictions, word_map, args.examples_per_category)
    report["word_map_used"] = bool(word_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("utterances", "wer", "error_types", "by_language",
                                                   "by_switch", "by_duration")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
