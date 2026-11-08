"""Map English words written in Devanagari to their Latin spelling, from training data only.

    python -m src.data.script_normalisation --config configs/data.yaml

MUCS training transcripts write some English words in Devanagari (फाइल) and
others in Latin (file), so the model has to learn two spellings for the same
sound. Blind references write English in Latin. This builds a map from
Devanagari spellings to Latin training words using two kinds of evidence, both
from the training split:

1. Sound. CMUdict pronunciations of Latin training words are spelled out in
   Devanagari by rule, and must match the Devanagari word, exactly or, for long
   words, closely.
2. Context. The two words must occur among the same neighbours. Contexts are
   scored again after mapping, so an English phrase written in Devanagari
   (स्पोकन ट्यूटोरियल) is compared with its Latin form (spoken tutorial).

Short words need stronger context evidence, because they collide with Hindi:
की ("of") sounds like "key". The map reads only train.jsonl. The blind word map,
hin2engwords.txt, is never used.

Writes data/processed/script_map/hi2en_train.json, results/script_map_report.json,
and normalised train and val manifests in data/processed/manifests_latin/. Test
and blind are copied unchanged, because they are scored against their own
references.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import re
import shutil
import unicodedata
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from src import ROOT
from src.analysis.error_analysis import script_of
from src.utils.config import data_path, load_config, read_manifest

HALANT = "्"
CONSONANTS = {
    "B": ["ब"], "CH": ["च"], "D": ["ड", "द"], "DH": ["द"], "F": ["फ"], "G": ["ग"],
    "HH": ["ह"], "JH": ["ज"], "K": ["क"], "L": ["ल"], "M": ["म"], "N": ["न"], "NG": ["ं"],
    "P": ["प"], "R": ["र"], "S": ["स"], "SH": ["श"], "T": ["ट", "त"], "TH": ["थ"], "V": ["व"],
    "W": ["व"], "Y": ["य"], "Z": ["ज"], "ZH": ["ज"],
}
# Each vowel as (independent letter, sign after a consonant); "" is the inherent vowel.
VOWELS = {
    "AA": [("आ", "ा"), ("ऑ", "ॉ"), ("ओ", "ो")], "AE": [("ऐ", "ै"), ("ए", "े")],
    "AH": [("अ", ""), ("आ", "ा"), ("इ", "ि")], "AO": [("ऑ", "ॉ"), ("ओ", "ो")],
    "AW": [("आउ", "ाउ")], "AY": [("आइ", "ाइ"), ("आय", "ाय")], "EH": [("ए", "े"), ("ऐ", "ै")],
    "ER": [("अर", "र")], "EY": [("ए", "े")], "IH": [("इ", "ि")], "IY": [("ई", "ी")],
    "OW": [("ओ", "ो")], "OY": [("ऑय", "ॉय")], "UH": [("उ", "ु")], "UW": [("ऊ", "ू"), ("यू", "्यू")],
}
GLIDE_AFTER = ("IY", "IH", "AY", "EY")  # रियल, डियो, डायल
SPELLING_LIMIT = 200
# Spelling key that ignores vowel length, nukta, chandrabindu, dental against
# retroflex t and d, and halant, all of which vary between transcribers.
FOLD = str.maketrans({"ॉ": "ो", "ै": "े", "ी": "ि", "ू": "ु", "ऑ": "ओ", "ऐ": "ए", "ई": "इ",
                      "ऊ": "उ", "ँ": "ं", "़": None, "ः": None, "द": "ड", "त": "ट", HALANT: None})

# Acceptance rules, set by reading the candidate lists of the training split.
MIN_LATIN_COUNT = 3       # a Latin word must be established in train
EXACT_CONTEXT = 0.30      # exact spelling match, any length
EXACT_LONG_CONTEXT = 0.10  # exact match, Latin word of 5+ letters
EXACT_LONGER_CONTEXT = 0.08  # exact match, Latin word of 6+ letters
CLOSE_DISTANCE = 0.20     # normalised edit distance of the folded spellings
CLOSE_CONTEXT = 0.40      # close match, Latin word of 5+ letters
WINDOW = 2
ROUNDS = 3
# Matches that pass the rules but are wrong, found by reading the accepted list.
EXCLUDED = {
    "आदर": "Hindi for respect, not 'other'",
    "गेम": "game, which is not a Latin training word; matched 'gamma'",
    "पोर्ट्स": "ports; matched 'parts'",
    "फेल्ड": "a misspelling of field; matched 'failed'",
    "स्टैचू": "unclear; matched 'statue'",
}


def fold(word: str) -> str:
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", word).translate(FOLD))


def spellings(phones: list[str], limit: int = SPELLING_LIMIT) -> set[str]:
    """Plausible Devanagari spellings of one CMUdict pronunciation."""
    phones = [re.sub(r"\d", "", phone) for phone in phones]
    options = []
    for i, phone in enumerate(phones):
        previous = phones[i - 1] if i else None
        following = phones[i + 1] if i + 1 < len(phones) else None
        if phone in CONSONANTS:
            joins = following in CONSONANTS and phone != "NG"
            options.append([letter + (HALANT if joins else "") for letter in CONSONANTS[phone]])
            continue
        after_consonant = previous in CONSONANTS and previous != "NG"
        after_vowel = previous in VOWELS
        choices = set()
        for independent, sign in VOWELS[phone]:
            if phone == "ER":
                tail = HALANT if following in CONSONANTS else ""
                choices.add(("र" if after_consonant else "यर" if after_vowel else "अर") + tail)
            elif after_consonant:
                choices.add(sign)
            elif after_vowel and previous in GLIDE_AFTER:
                choices.add("य" + sign)
                if phone == "AH":
                    choices.add("")
            elif after_vowel and phone == "AH":
                choices.add("")
            else:
                choices.add(independent)
        options.append(sorted(choices))
    return {unicodedata.normalize("NFC", "".join(combo))
            for combo in itertools.islice(itertools.product(*options), limit)}


def load_cmudict(path: Path) -> dict[str, list[list[str]]]:
    entries: dict[str, list[list[str]]] = collections.defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            word, *phones = line.split()
            entries[re.sub(r"\(\d+\)$", "", word)].append(phones)
    return entries


def pronunciations(word: str, cmudict: dict[str, list[list[str]]]) -> list[list[str]]:
    """CMUdict entries, or two joined entries for compounds such as backspace."""
    if word in cmudict:
        return cmudict[word][:3]
    for cut in range(3, len(word) - 2):
        left, right = word[:cut], word[cut:]
        if left in cmudict and right in cmudict:
            return [cmudict[left][0] + cmudict[right][0]]
    return []


def keeps_plural(devanagari: str, latin: str) -> bool:
    """A Devanagari plural in -्स must map to a Latin plural. Words such as मैसेज
    end in ज without being plural, so only this one ending is checked."""
    return latin.endswith("s") or not devanagari.endswith("्स")


def context_vectors(sentences: list[list[str]], mapping: dict[str, str]) -> dict[str, collections.Counter]:
    """Neighbours of each word, with neighbours (but not the word itself) mapped."""
    vectors: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for words in sentences:
        mapped = [mapping.get(word, word) for word in words]
        for i, word in enumerate(words):
            for j in range(max(0, i - WINDOW), min(len(words), i + WINDOW + 1)):
                if j != i:
                    vectors[word][(j - i, mapped[j])] += 1
    return vectors


def cosine(a: collections.Counter, b: collections.Counter) -> float:
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[key] * b[key] for key in shared)
    return dot / math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))


def accepted(latin: str, distance: float, context: float) -> bool:
    if distance == 0:
        return (context >= EXACT_CONTEXT
                or (len(latin) >= 5 and context >= EXACT_LONG_CONTEXT)
                or (len(latin) >= 6 and context >= EXACT_LONGER_CONTEXT))
    return len(latin) >= 5 and context >= CLOSE_CONTEXT


def candidate_pairs(counts: collections.Counter, cmudict) -> list[tuple[str, str, float]]:
    latin = [w for w, c in counts.items()
             if script_of(w) == "english_script" and c >= MIN_LATIN_COUNT and len(w) >= 2 and w.isalpha()]
    index: dict[str, set[str]] = collections.defaultdict(set)
    for word in latin:
        for pronunciation in pronunciations(word, cmudict):
            for spelling in spellings(pronunciation):
                index[fold(spelling)].add(word)
    keys = list(index)
    pairs = []
    for word in counts:
        if script_of(word) != "hindi_script" or word in EXCLUDED or re.search(r"\d", word):
            continue
        key = fold(word)
        found = {(latin_word, 0.0) for latin_word in index.get(key, ())}
        if not found and len(key) >= 4:
            for other in keys:
                if abs(len(other) - len(key)) <= 2:
                    distance = Levenshtein.normalized_distance(key, other)
                    if distance <= CLOSE_DISTANCE:
                        found |= {(latin_word, distance) for latin_word in index[other]}
        pairs.extend((word, latin_word, distance) for latin_word, distance in found
                     if keeps_plural(word, latin_word))
    return pairs


def build_map(sentences: list[list[str]], cmudict) -> tuple[dict[str, str], dict[str, float]]:
    counts = collections.Counter(word for words in sentences for word in words)
    pairs = candidate_pairs(counts, cmudict)
    mapping: dict[str, str] = {}
    best: dict[str, tuple[str, float]] = {}
    for _ in range(ROUNDS):
        vectors = context_vectors(sentences, mapping)
        best = {}
        for word, latin_word, distance in pairs:
            context = cosine(vectors[word], vectors[latin_word])
            if accepted(latin_word, distance, context) and context > best.get(word, ("", -1.0))[1]:
                best[word] = (latin_word, context)
        mapping = {word: latin_word for word, (latin_word, _) in best.items()}
    return mapping, {word: round(context, 3) for word, (_, context) in best.items()}


def normalise(text: str, mapping: dict[str, str]) -> str:
    return " ".join(mapping.get(word, word) for word in text.split())


def write_manifests(mapping: dict[str, str], source: Path, output: Path) -> dict[str, int]:
    """Normalised train and val; test and blind copied, since they keep their references."""
    output.mkdir(parents=True, exist_ok=True)
    changed = {}
    for split in ("train", "val"):
        rows = read_manifest(source / f"{split}.jsonl")
        count = 0
        with (output / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                text = normalise(row["text"], mapping)
                count += sum(a != b for a, b in zip(row["text"].split(), text.split()))
                row = {**row, "text_unnormalised": row["text"], "text": text,
                       "text_raw": normalise(row["text_raw"], mapping)}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        changed[split] = count
    for split in ("test", "blind"):
        shutil.copy2(source / f"{split}.jsonl", output / f"{split}.jsonl")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--cmudict", default="data/external/cmudict/cmudict.dict")
    parser.add_argument("--output", default="data/processed/manifests_latin")
    args = parser.parse_args()
    config = load_config(args.config)
    source = data_path(config["data"]["manifest_dir"])

    train = read_manifest(source / "train.jsonl")
    if any(row.get("split", "train") != "train" for row in train):
        raise ValueError("the script map must be built from training rows only")
    sentences = [row["text"].split() for row in train]
    counts = collections.Counter(word for words in sentences for word in words)
    mapping, contexts = build_map(sentences, load_cmudict(data_path(args.cmudict)))

    map_dir = data_path("data/processed/script_map")
    map_dir.mkdir(parents=True, exist_ok=True)
    (map_dir / "hi2en_train.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=0), encoding="utf-8")
    changed = write_manifests(mapping, source, data_path(args.output))

    total = sum(counts.values())
    tokens = sum(counts[word] for word in mapping)
    report = {
        "source": "train.jsonl and CMUdict only; hin2engwords.txt not used",
        "devanagari_words_mapped": len(mapping),
        "train_tokens_changed": tokens,
        "train_token_percent": round(100 * tokens / total, 2),
        "tokens_changed_by_split": changed,
        "rules": {"min_latin_count": MIN_LATIN_COUNT, "exact_context": EXACT_CONTEXT,
                  "exact_long_context": EXACT_LONG_CONTEXT, "exact_longer_context": EXACT_LONGER_CONTEXT,
                  "close_distance": CLOSE_DISTANCE, "close_context": CLOSE_CONTEXT,
                  "window": WINDOW, "rounds": ROUNDS},
        "excluded_by_hand": EXCLUDED,
        "mapping": [{"devanagari": word, "latin": mapping[word], "train_count": counts[word],
                     "context_similarity": contexts[word]}
                    for word in sorted(mapping, key=lambda w: -counts[w])],
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "script_map_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("devanagari_words_mapped", "train_tokens_changed", "train_token_percent",
                       "tokens_changed_by_split")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
