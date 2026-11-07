"""Measure GPU memory and speed at real worst-case batch sizes.

    python -m src.training.profile_gpu --config configs/train_baseline.yaml

Each batch uses the longest real training utterances, with mixed precision and
gradient checkpointing, and with the encoder unfrozen, since that is the most
expensive training state. Writes results/gpu_profile.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor

from src import ROOT
from src.data.dataset import Collator
from src.models.ctc_model import build_model, set_encoder_frozen
from src.training.train import GRAD_SCALE_INIT, build_dataset, to_device
from src.utils.config import data_path, load_config, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[160_000, 240_000, 320_000, 400_000, 480_000, 640_000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="results/gpu_profile.json")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["training"]["seed"])

    vocab_path = data_path(config["data"]["tokenizer_dir"]) / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    dataset = build_dataset(config, "train", vocab)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(config["model"]["pretrained"])
    collate = Collator(extractor)
    order = np.argsort(-dataset.samples)

    device = torch.device("cuda")
    model, _ = build_model(config["model"], vocab)
    model.to(device).train()
    set_encoder_frozen(model, False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    scaler = torch.amp.GradScaler("cuda", init_scale=GRAD_SCALE_INIT)

    total, reserved_by_others = torch.cuda.mem_get_info()[1], None
    free_before = torch.cuda.mem_get_info()[0]
    results = []
    for size in args.sizes:
        longest = int(dataset.samples[order[0]])
        count = max(size // longest, 1)
        chosen = order[:count].tolist()
        batch = to_device(collate([dataset[i] for i in chosen]), device)
        padded = batch["input_values"].numel()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        times, failure = [], None
        try:
            for _ in range(args.repeats):
                began = time.time()
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(**batch).loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                times.append(time.time() - began)
        except torch.cuda.OutOfMemoryError as error:
            failure = str(error).splitlines()[0][:120]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        step = float(np.median(times[1:])) if len(times) > 1 else None
        row = {
            "target_samples": size,
            "padded_samples": padded,
            "utterances": count,
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20),
            "seconds_per_step": round(step, 3) if step else None,
            "audio_seconds_per_second": round(padded / 16000 / step, 1) if step else None,
            "out_of_memory": failure,
        }
        print(row, flush=True)
        results.append(row)

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_mb": round(total / 2**20),
        "gpu_free_at_start_mb": round(free_before / 2**20),
        "mixed_precision": True,
        "gradient_checkpointing": bool(config["model"].get("gradient_checkpointing")),
        "encoder_frozen": False,
        "results": results,
    }
    output = ROOT / args.output
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
