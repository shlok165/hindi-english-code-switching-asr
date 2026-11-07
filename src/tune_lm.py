"""Tune the LM weight and word insertion score on validation data only.

    python -m src.tune_lm --checkpoint data/checkpoints/baseline_him4200/best

Decodes a fixed subset of val for every grid point at a modest beam width, keeps
the best settings, then tests wider beams with them. Every result is written to
results/lm_tuning.csv. This command cannot read test or blind data: it only ever
opens the val cache and val references.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

from src import ROOT
from src.beam_search import BeamPool, BeamSettings
from src.metrics.wer import score
from src.utils.config import data_path, read_manifest

ALPHAS = (0.0, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5)
BETAS = (-1.0, 0.0, 1.0, 2.0, 3.0)
UNK_OFFSETS = (-5.0, -10.0, -20.0)
BEAM_WIDTHS = (50, 100, 200, 500)


def evenly_spaced(count: int, size: int) -> list[int]:
    return list(range(0, count, max(count // size, 1)))[:size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--kenlm", default="data/language_model/train_5gram.klm")
    parser.add_argument("--subset", type=int, default=500)
    parser.add_argument("--grid-beam", type=int, default=50)
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 4) - 2, 1))
    args = parser.parse_args()

    cache = data_path("data/processed/logits") / args.checkpoint.parent.name / args.checkpoint.name
    vocab = json.loads((args.checkpoint / "vocab.json").read_text(encoding="utf-8"))
    entries = read_manifest(data_path("data/processed/manifests") / "val.jsonl")
    positions = [p for p in evenly_spaced(len(entries), args.subset) if entries[p]["text"]]
    references = [entries[p]["text"] for p in positions]
    kenlm = data_path(args.kenlm)

    rows = []
    output = ROOT / "results" / "lm_tuning.csv"
    with BeamPool(cache, "val", vocab, kenlm, args.workers) as pool:
        def run(stage: str, chosen: list[int], truth: list[str], settings: BeamSettings) -> float:
            began = time.time()
            hypotheses = pool.decode(chosen, settings)
            wer = score(zip(truth, hypotheses)).wer
            row = {"stage": stage, "utterances": len(chosen), **asdict(settings),
                   "val_wer": round(wer, 3), "seconds": round(time.time() - began, 1)}
            rows.append(row)
            print(row, flush=True)
            with output.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, list(row))
                writer.writeheader()
                writer.writerows(rows)
            return wer

        base = BeamSettings(beam_width=args.grid_beam)
        best, best_wer = base, float("inf")
        for alpha, beta in itertools.product(ALPHAS, BETAS):
            candidate = replace(base, alpha=alpha, beta=beta)
            wer = run("alpha_beta", positions, references, candidate)
            if wer < best_wer:
                best, best_wer = candidate, wer
        for offset in UNK_OFFSETS:
            candidate = replace(best, unk_score_offset=offset)
            wer = run("unk_offset", positions, references, candidate)
            if wer < best_wer:
                best, best_wer = candidate, wer
        for width in BEAM_WIDTHS:
            run("beam_width", positions, references, replace(best, beam_width=width))

    print(json.dumps({"best": asdict(best), "subset_val_wer": round(best_wer, 3)}, indent=2))


if __name__ == "__main__":
    main()
