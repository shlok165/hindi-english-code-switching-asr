"""Time real training updates under different batching settings.

    python -m src.training.benchmark_batching --config configs/train_baseline.yaml \
        --setting 0:1:8 --setting 960000:0:1

Each setting is max_samples:max_utterances:accumulation, with 0 meaning no cap
on utterances. Batches come from the real shuffled training order, so speed and
memory reflect training rather than the worst case that profile_gpu.py measures.
Writes results/batching_benchmark.json.
"""

from __future__ import annotations

import argparse
import json
import time

import src  # noqa: F401
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor

from src import ROOT
from src.data.dataset import LengthBatchSampler
from src.models.ctc_model import build_model
from src.training.train import GRAD_SCALE_INIT, build_dataset, make_loader, to_device
from src.utils.config import data_path, load_config, seed_everything


def run_setting(config, train_set, extractor, vocab, spec: str, updates: int, warmup: int) -> dict:
    max_samples, max_utterances, accumulate = (int(part) for part in spec.split(":"))
    max_samples = max_samples or config["training"]["max_samples_per_batch"]
    sampler = LengthBatchSampler(train_set.samples, max_samples, config["training"]["seed"],
                                 max_utterances=max_utterances or None)
    loader = make_loader(train_set, sampler, extractor, 0)
    device = torch.device("cuda")
    model, _ = build_model(config["model"], vocab)
    model.to(device).train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    scaler = torch.amp.GradScaler("cuda", init_scale=GRAD_SCALE_INIT)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    times, audio, padded, utterances = [], 0.0, 0, 0
    batches, done, began = 0, 0, time.time()
    for batch in loader:
        inputs = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**inputs).loss / accumulate
        scaler.scale(loss).backward()
        batches += 1
        if done >= warmup:
            mask = inputs["attention_mask"]
            audio += float(mask.sum()) / 16000
            padded += mask.numel()
            utterances += mask.shape[0]
        if batches % accumulate:
            continue
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        done += 1
        now = time.time()
        if done > warmup:
            times.append(now - began)
        began = now
        if done >= warmup + updates:
            break
    seconds = float(np.sum(times))
    row = {
        "setting": spec,
        "max_samples": max_samples,
        "max_utterances": max_utterances or None,
        "accumulation": accumulate,
        "updates_timed": len(times),
        "seconds_per_update": round(seconds / len(times), 3),
        "utterances_per_update": round(utterances / len(times), 2),
        "audio_seconds_per_update": round(audio / len(times), 1),
        "audio_seconds_per_second": round(audio / seconds, 1),
        "padding_percent": round(100 * (1 - audio * 16000 / max(padded, 1)), 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20),
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--setting", action="append", required=True)
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", default="results/batching_benchmark.json")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["training"]["seed"])
    if config["training"].get("gpu_memory_fraction"):
        torch.cuda.set_per_process_memory_fraction(config["training"]["gpu_memory_fraction"])
    vocab = json.loads((data_path(config["data"]["tokenizer_dir"]) / "vocab.json").read_text(encoding="utf-8"))
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(config["model"]["pretrained"])
    extractor.return_attention_mask = True
    train_set = build_dataset(config, "train", vocab)

    rows = []
    for spec in args.setting:
        row = run_setting(config, train_set, extractor, vocab, spec, args.updates, args.warmup)
        print(row, flush=True)
        rows.append(row)
    report = {"gpu": torch.cuda.get_device_name(0), "config": args.config, "results": rows}
    output = ROOT / args.output
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
