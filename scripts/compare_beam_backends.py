"""Decode the same log-probabilities with real KenLM and with the ARPA backend; they must agree.

    python scripts/compare_beam_backends.py --logprobs results/cpu_eval/lm_check/beam_inputs/tts_en_logprobs.npy \
        --vocab results/cpu_eval/lm_check/beam_inputs/vocab.json --decode-config configs/decode_iteration4.yaml \
        --output results/cpu_eval/lm_check/beam_equivalence.json

Needs kenlm and pyctcdecode, but not torch: the log-probabilities come from a
file (src.inference.log_probabilities saved with numpy). Besides the saved
array it decodes perturbed copies of it (noise added, frames dropped, frames
repeated) so that many different word sequences compete, and reports every
case where the two backends disagree. Written to be run where kenlm installs,
such as WSL, against the repository on /mnt/c.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.beam_search import BeamSettings, beam_decode, build_beam_decoder  # noqa: E402
from src.utils.config import data_path, load_config  # noqa: E402


def settings_from_config(path: str) -> tuple[BeamSettings, str]:
    """As src.inference.settings_from_config, without importing torch."""
    decoder = load_config(path)["decoder"]
    fields = BeamSettings.__dataclass_fields__
    return BeamSettings(**{k: v for k, v in decoder.items() if k in fields}), decoder["kenlm_model"]


def variants(log_probs: np.ndarray, count: int, seed: int) -> list[tuple[str, np.ndarray]]:
    generator = np.random.default_rng(seed)
    out = [("original", log_probs)]
    for i in range(count):
        noisy = log_probs + generator.normal(0, 1.5, size=log_probs.shape).astype(np.float32)
        out.append((f"noise_{i}", noisy))
        keep = np.sort(generator.choice(len(log_probs), size=len(log_probs) * 2 // 3, replace=False))
        out.append((f"dropped_{i}", noisy[keep]))
        repeat = np.repeat(np.arange(len(log_probs)), generator.integers(1, 3, size=len(log_probs)))
        out.append((f"repeated_{i}", noisy[repeat]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logprobs", required=True, type=Path)
    parser.add_argument("--vocab", required=True, type=Path)
    parser.add_argument("--decode-config", default="configs/decode_iteration4.yaml")
    parser.add_argument("--kenlm", type=Path, help="override the .klm named in the decode config")
    parser.add_argument("--variants", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    settings, configured = settings_from_config(args.decode_config)
    kenlm_path = args.kenlm or data_path(configured)
    vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
    log_probs = np.load(args.logprobs).astype(np.float32)

    timings = {}
    decoders = {}
    for backend in ("kenlm", "arpa"):
        started = time.perf_counter()
        decoders[backend] = build_beam_decoder(vocab, kenlm_path, settings, backend)
        timings[f"{backend}_load_seconds"] = round(time.perf_counter() - started, 2)

    rows, disagreements = [], []
    for name, scores in variants(log_probs, args.variants, args.seed):
        outputs, seconds = {}, {}
        for backend, decoder in decoders.items():
            started = time.perf_counter()
            outputs[backend] = beam_decode(scores, vocab, decoder, settings)
            seconds[backend] = round(time.perf_counter() - started, 3)
        row = {"variant": name, "frames": int(len(scores)), "kenlm": outputs["kenlm"], "arpa": outputs["arpa"],
               "agree": outputs["kenlm"] == outputs["arpa"], "seconds": seconds}
        rows.append(row)
        if not row["agree"]:
            disagreements.append(row)
    report = {
        "logprobs": str(args.logprobs), "decode_config": args.decode_config, "settings": settings.__dict__,
        "language_model": str(kenlm_path), "cases": len(rows), "disagreements": len(disagreements),
        "mean_seconds": {b: round(float(np.mean([r["seconds"][b] for r in rows])), 3) for b in ("kenlm", "arpa")},
        **timings, "original": rows[0], "disagreeing_cases": disagreements, "all_cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("all_cases", "disagreeing_cases")},
                     ensure_ascii=False, indent=2))
    sys.exit(1 if disagreements else 0)


if __name__ == "__main__":
    main()
