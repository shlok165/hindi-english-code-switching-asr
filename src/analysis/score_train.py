"""Score every training utterance with a trained model, to find misaligned segments.

    python -m src.analysis.score_train --config configs/train_baseline.yaml \
        --checkpoint data/checkpoints/baseline_rtx3050/best --device cpu

MUCS segments sometimes cut speech: the transcript holds words the audio does not,
or the other way round. A model that is good on clean segments fails badly on
those, so each utterance gets its greedy CER against its transcript, the CTC loss
per transcript character, and the ratio of output to transcript length. Writes
data/processed/train_scores/<run>_<checkpoint>.jsonl, one row per utterance, and
resumes from that file if it is interrupted. Only the training split is read.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import src  # noqa: F401
import torch
import torch.nn.functional as F

from src.beam_search import greedy_from_log_probs
from src.inference import log_probabilities
from src.metrics.cer import score as score_cer
from src.models.ctc_model import load_model
from src.models.tokenizer import BLANK
from src.training.train import build_dataset
from src.utils.config import data_path, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="score only this many utterances (for testing)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    checkpoint = data_path(args.checkpoint)
    config = load_config(args.config)

    model, vocab, extractor = load_model(checkpoint)
    device = torch.device(args.device)
    model.to(device).eval()
    dataset = build_dataset(config, "train", vocab)
    blank = vocab[BLANK]

    output = data_path("data/processed/train_scores") / f"{checkpoint.parent.name}_{checkpoint.name}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            done = {json.loads(line)["id"] for line in handle}
    began, scored = time.time(), 0
    with output.open("a", encoding="utf-8", newline="\n") as handle, torch.no_grad():
        for index in range(min(len(dataset), args.limit or len(dataset))):
            item = dataset[index]
            if item["id"] in done:
                continue
            scores = log_probabilities(item["audio"], model, extractor, device)
            labels = torch.tensor(item["labels"], dtype=torch.long)
            loss = F.ctc_loss(scores.unsqueeze(1).cpu(), labels.unsqueeze(0), torch.tensor([len(scores)]),
                              torch.tensor([len(labels)]), blank=blank, reduction="sum", zero_infinity=True)
            hypothesis = greedy_from_log_probs(scores.cpu().numpy(), vocab)
            reference = item["text"]
            handle.write(json.dumps({
                "id": item["id"],
                "duration": round(len(item["audio"]) / 16000, 2),
                "reference_chars": len(reference),
                "loss_per_char": round(float(loss) / max(len(labels), 1), 4),
                "cer": round(score_cer([(reference, hypothesis)], clean=False).wer, 2),
                "length_ratio": round(len(hypothesis) / max(len(reference), 1), 3),
                "hypothesis": hypothesis,
            }, ensure_ascii=False) + "\n")
            scored += 1
            if scored % 1000 == 0:
                handle.flush()
                rate = (time.time() - began) / scored
                print(f"{len(done) + scored}/{len(dataset)} scored, {rate:.3f} s each", flush=True)
    print(f"done: {output}")


if __name__ == "__main__":
    main()
