"""Tune the LM weight and word insertion score on validation data only.

    python -m src.tune_lm --checkpoint data/checkpoints/baseline_rtx3050/best

Decodes val (all of it by default) for every grid point at a modest beam width,
refines the LM weight around the winner, then tests wider beams with it. Every
result is written to results/lm_tuning.csv. This command cannot read test or blind data: it only ever
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
UNK_OFFSETS = (-2.0, -3.0, -4.0, -5.0, -10.0, -20.0)  # iteration 1 tuned -5, -10, -20, then checked -4 to 0
BEAM_WIDTHS = (50, 100, 200, 500, 1000, 3000)  # 3000 is the width the MUCS baseline used
REFINE_STEPS = (-0.1, -0.05, 0.05, 0.1)  # finer alpha steps tried around the coarse winner
REFINE_BETA_STEPS = (-0.5, 0.5)
WIDTH_TOLERANCE = 0.05  # WER points a wider beam must gain to justify its cost

CONFIG_TEMPLATE = """# Tuned on validation data by src.tune_lm, with test and blind untouched.
# WER on the tuning utterances of val: {wer:.3f}. Every grid result is in {results}.
decoder:
  kenlm_model: {kenlm}
  beam_width: {beam_width}
  alpha: {alpha}               # LM weight
  beta: {beta}                # Word insertion score
  unk_score_offset: {unk_score_offset}  # Penalty for words outside the LM vocabulary
  beam_prune_logp: {beam_prune_logp}   # pyctcdecode default
  token_min_logp: {token_min_logp}     # pyctcdecode default
"""


def evenly_spaced(count: int, size: int) -> list[int]:
    return list(range(0, count, max(count // size, 1)))[:size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--kenlm", default="data/language_model/train_5gram.klm")
    parser.add_argument("--subset", type=int, default=0, help="utterances to tune on; 0 means all of val")
    parser.add_argument("--grid-beam", type=int, default=50)
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 4) - 2, 1))
    parser.add_argument("--write-config", action="store_true", help="save the winner to --config-out")
    parser.add_argument("--config-out", default="configs/decode.yaml")
    parser.add_argument("--results", default="results/lm_tuning.csv", help="where every grid result is written")
    parser.add_argument("--manifest-dir", default="data/processed/manifests",
                        help="where val.jsonl is read from, such as the script-normalised manifests")
    args = parser.parse_args()
    args.checkpoint = data_path(args.checkpoint)

    cache = data_path("data/processed/logits") / args.checkpoint.parent.name / args.checkpoint.name
    vocab = json.loads((args.checkpoint / "vocab.json").read_text(encoding="utf-8"))
    entries = read_manifest(data_path(args.manifest_dir) / "val.jsonl")
    size = args.subset or len(entries)
    positions = [p for p in evenly_spaced(len(entries), size) if entries[p]["text"]]
    references = [entries[p]["text"] for p in positions]
    kenlm = data_path(args.kenlm)

    rows = []
    output = ROOT / args.results
    output.parent.mkdir(parents=True, exist_ok=True)
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
        coarse = best
        for alpha_step, beta_step in itertools.product(REFINE_STEPS + (0.0,), REFINE_BETA_STEPS + (0.0,)):
            alpha = round(coarse.alpha + alpha_step, 3)
            if alpha < 0 or (alpha_step == 0 and beta_step == 0):
                continue
            candidate = replace(coarse, alpha=alpha, beta=coarse.beta + beta_step)
            wer = run("refine", positions, references, candidate)
            if wer < best_wer:
                best, best_wer = candidate, wer
        for offset in UNK_OFFSETS:
            candidate = replace(best, unk_score_offset=offset)
            wer = run("unk_offset", positions, references, candidate)
            if wer < best_wer:
                best, best_wer = candidate, wer
        for width in BEAM_WIDTHS:
            wer = run("beam_width", positions, references, replace(best, beam_width=width))
            if wer < best_wer - WIDTH_TOLERANCE:
                best, best_wer = replace(best, beam_width=width), wer

    if args.write_config:
        text = CONFIG_TEMPLATE.format(wer=best_wer, kenlm=args.kenlm, results=args.results, **asdict(best))
        (ROOT / args.config_out).write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({"best": asdict(best), "subset_val_wer": round(best_wer, 3)}, indent=2))


if __name__ == "__main__":
    main()
