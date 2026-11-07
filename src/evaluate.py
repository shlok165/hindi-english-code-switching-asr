"""Decode a split from cached log-probabilities and score the result.

    python -m src.evaluate --split val --decoder greedy --checkpoint data/checkpoints/baseline_rtx3050/best
    python -m src.evaluate --split blind --decoder beam --checkpoint data/checkpoints/baseline_rtx3050/best

Metrics written, and what each one is:
- `wer`, `cer`: scored against the raw references with the organisers' rules, so
  on blind `wer` is the official metric. CER is ours, on the same strings.
- `twer`: official, blind only. Confirmed by re-running compute_wer.py.
- `twer_approximate`: test only. Not the official metric; see src/metrics/official.py.
- `wer_cleaned`, `cer_cleaned`: against training-style cleaned text, for checks.

Val gets no T-WER, because the word map comes from the blind references and val
results drive tuning.

Keep heavy imports out of module level. Windows starts each decoding worker by
re-importing this file, so a torch import here would cost every worker about a
gigabyte of memory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from src import ROOT
from src.beam_search import BeamPool, BeamSettings, LogProbCache, greedy_from_log_probs
from src.metrics.cer import score as score_cer
from src.metrics.official import (
    INFERENCE_DIR,
    approximate_twer,
    load_maps,
    normalise_hypothesis,
    normalise_reference,
    official_wer,
)
from src.metrics.wer import score as score_wer
from src.utils.config import data_path, load_config, read_manifest

BLIND_REFERENCES = "data/interim/extracted/blind/subtask2_blindtest_wReadme/hindi/files/text"


def beam_settings(decode_config: str) -> tuple[BeamSettings, Path]:
    decoder = load_config(decode_config)["decoder"]
    fields = BeamSettings.__dataclass_fields__
    settings = BeamSettings(**{key: value for key, value in decoder.items() if key in fields})
    return settings, data_path(decoder["kenlm_model"])


def run_official_script(hypotheses: Path) -> dict[str, float]:
    """Run the organisers' own scorer, unchanged, on a hypothesis file."""
    root = data_path(INFERENCE_DIR)
    result = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", str(root / "compute_wer.py"),
         "--reffile", str(data_path(BLIND_REFERENCES)), "--hypfile", str(hypotheses),
         "--punctmappingfile", str(root / "punct.mapping.txt"),
         "--wordmappingfile", str(root / "hin2engwords.txt")],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    values = re.findall(r"WER: ([0-9.]+)", result.stdout)
    return {"wer": float(values[0]), "twer": float(values[1])}


def compute_metrics(split: str, entries: list[dict], hypotheses: list[str], hyp_path: Path) -> dict[str, Any]:
    maps = load_maps()
    raw_pairs = [(e["text_raw"], h) for e, h in zip(entries, hypotheses)]
    scored = [(normalise_reference(r), normalise_hypothesis(h, maps.punctuation))
              for r, h in raw_pairs if normalise_reference(r)]
    cleaned = [(e["text"], h) for e, h in zip(entries, hypotheses) if e["text"]]
    official = official_wer(raw_pairs, maps)
    metrics: dict[str, Any] = {
        "wer": official["wer"],
        "cer": score_cer(scored, clean=False).wer,
        "wer_cleaned": score_wer(cleaned).wer,
        "cer_cleaned": score_cer(cleaned).wer,
        "reference_words": official["reference_words"],
        "scored_utterances": official["scored_utterances"],
        "empty_references": len(entries) - official["scored_utterances"],
        "empty_hypotheses": sum(not h for h in hypotheses),
    }
    if split == "blind":
        metrics["twer"] = official["twer"]
        check = run_official_script(hyp_path)
        metrics["official_script"] = check
        if abs(check["wer"] - metrics["wer"]) > 0.01 or abs(check["twer"] - metrics["twer"]) > 0.01:
            raise RuntimeError(f"disagrees with compute_wer.py: {check} vs {metrics}")
    elif split == "test":
        metrics.update(approximate_twer(raw_pairs, maps))
    return metrics


def evaluate(
    split: str,
    checkpoint: Path,
    decoder_name: str,
    decode_config: str,
    output_dir: Path,
    workers: int,
) -> dict[str, Any]:
    entries = read_manifest(data_path("data/processed/manifests") / f"{split}.jsonl")
    ids = [entry["id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{split} manifest contains duplicate utterance ids")

    cache_dir = data_path("data/processed/logits") / checkpoint.parent.name / checkpoint.name
    if not (cache_dir / f"{split}.npy").exists():
        from src.logit_cache import build_cache

        build_cache(split, checkpoint, cache_dir)
    cache = LogProbCache(cache_dir, split)
    if [row["id"] for row in cache.index] != ids:
        raise ValueError("cached log-probabilities do not match the manifest order")
    vocab = json.loads((checkpoint / "vocab.json").read_text(encoding="utf-8"))

    settings: BeamSettings | None = None
    if decoder_name == "greedy":
        hypotheses = [greedy_from_log_probs(cache[i], vocab) for i in range(len(cache))]
    else:
        settings, kenlm = beam_settings(decode_config)
        with BeamPool(cache_dir, split, vocab, kenlm, workers) as pool:
            hypotheses = pool.decode(range(len(cache)), settings)

    output_dir.mkdir(parents=True, exist_ok=True)
    hyp_path = output_dir / f"{split}_{decoder_name}.hyp"
    with hyp_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(f"{i} {h}\n" for i, h in zip(ids, hypotheses))

    maps = load_maps()
    with (output_dir / f"{split}_{decoder_name}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for entry, hypothesis in zip(entries, hypotheses):
            handle.write(json.dumps({
                "id": entry["id"],
                "split": split,
                "audio": entry["audio"],
                "start": entry["start"],
                "end": entry["end"],
                "duration": entry["duration"],
                "reference": normalise_reference(entry["text_raw"]),
                "reference_cleaned": entry["text"],
                "hypothesis": normalise_hypothesis(hypothesis, maps.punctuation),
                "hypothesis_raw": hypothesis,
            }, ensure_ascii=False) + "\n")

    summary = {
        "split": split,
        "decoder": decoder_name,
        "alignment": "UnA",
        "checkpoint": checkpoint.as_posix().split("data/")[-1],
        "utterances": len(entries),
        "settings": settings.__dict__ if settings else None,
        **compute_metrics(split, entries, hypotheses, hyp_path),
    }
    path = output_dir / f"{split}_{decoder_name}_metrics.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("val", "test", "blind"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--decoder", default="greedy", choices=("greedy", "beam"))
    parser.add_argument("--decode-config", default="configs/decode.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "evaluations")
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 4) - 2, 1))
    args = parser.parse_args()
    evaluate(args.split, data_path(args.checkpoint), args.decoder, args.decode_config, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
