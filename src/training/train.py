"""CTC fine-tuning loop: mixed precision, accumulation, checkpoints and resume.

    python -m src.training.train --config configs/train_baseline.yaml
    python -m src.training.train --config configs/train_baseline.yaml --resume auto
    python -m src.training.train --config configs/train_smoke.yaml --set training.seed=7

Only the train and val manifests can be used here. Test and blind data are
refused, so they cannot affect training, tuning or checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor

from src.data.dataset import TRAINABLE_SPLITS, Collator, LengthBatchSampler, MucsDataset
from src.metrics.wer import score
from src.models.ctc_model import (
    build_model,
    greedy_decode,
    output_lengths,
    replace_directory,
    save_model,
    set_encoder_frozen,
)
from src.utils.config import data_path, load_config, run_metadata, seed_everything

# The default 65536 overflows fp16 CTC gradients and skips the first few updates.
GRAD_SCALE_INIT = 1024.0
LOG_FIELDS = ["update", "epoch", "train_loss", "val_loss", "val_wer", "lr", "grad_norm",
              "seconds", "peak_gpu_mb", "skipped_batches"]


def tri_stage(update: int, total: int, warmup: float, hold: float) -> float:
    """Learning-rate multiplier: linear up, flat, then cosine down to zero."""
    warmup_updates = max(int(total * warmup), 1)
    hold_end = warmup_updates + int(total * hold)
    if update < warmup_updates:
        return (update + 1) / warmup_updates
    if update < hold_end:
        return 1.0
    progress = (update - hold_end) / max(total - hold_end, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def manifest_path(config: dict, split: str) -> Path:
    if split not in TRAINABLE_SPLITS:
        raise ValueError(f"split {split!r} may not be used for training or tuning")
    return data_path(config["data"]["manifest_dir"]) / f"{split}.jsonl"


def build_dataset(config: dict, split: str, vocab: dict[str, int]) -> MucsDataset:
    data = config["data"]
    limit = data.get(f"limit_{split}")
    return MucsDataset(
        manifest_path(config, split), vocab, data["min_seconds"], data["max_seconds"], limit
    )


def make_loader(dataset: MucsDataset, sampler: LengthBatchSampler, extractor, workers: int) -> DataLoader:
    # An explicit generator stops the loader drawing from the global RNG, which
    # would make dropout and SpecAugment depend on when the iterator was created.
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=Collator(extractor),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(0),
    )


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("input_values", "attention_mask", "labels")
    return {key: batch[key].to(device, non_blocking=True) for key in keys}


@torch.no_grad()
def evaluate(model, loader, vocab, device, use_amp: bool) -> dict[str, float]:
    """Validation loss and greedy WER on normalised transcripts.

    This WER selects checkpoints. It is not the official challenge score.
    """
    model.eval()
    loss_sum, weight, pairs = 0.0, 0, []
    for batch in loader:
        inputs = to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            output = model(**inputs)
        size = inputs["labels"].shape[0]
        loss_sum += output.loss.item() * size
        weight += size
        lengths = output_lengths(model, inputs["attention_mask"].sum(-1))
        pairs.extend(zip(batch["texts"], greedy_decode(output.logits, lengths, vocab)))
    model.train()
    return {"val_loss": loss_sum / max(weight, 1), "val_wer": score(pairs).wer}


def rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(directory, model, vocab, extractor, optimizer, scheduler, scaler, position) -> None:
    temporary = directory.with_name(directory.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    save_model(temporary, model, vocab, extractor)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": rng_state(),
            "position": position,
        },
        temporary / "trainer_state.pt",
    )
    replace_directory(temporary, directory)


def latest_checkpoint(run_dir: Path) -> Path | None:
    steps = sorted(run_dir.glob("step-*[0-9]"), key=lambda p: int(p.name.split("-")[1]))
    return steps[-1] if steps else None


def prune_checkpoints(run_dir: Path, keep: int) -> None:
    steps = sorted(run_dir.glob("step-*[0-9]"), key=lambda p: int(p.name.split("-")[1]))
    for old in steps[:-keep]:
        shutil.rmtree(old)


def train(config: dict[str, Any], resume: str | None = None) -> dict[str, Any]:
    settings = config["training"]
    seed_everything(settings["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(settings["mixed_precision"]) and device.type == "cuda"

    tokenizer_dir = data_path(config["data"]["tokenizer_dir"])
    vocab = json.loads((tokenizer_dir / "vocab.json").read_text(encoding="utf-8"))
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(config["model"]["pretrained"])
    extractor.return_attention_mask = True

    train_set = build_dataset(config, "train", vocab)
    val_set = build_dataset(config, "val", vocab)
    if not len(train_set) or not len(val_set):
        raise ValueError("train or val manifest is empty after filtering")
    print(f"train {len(train_set)} utts (dropped {train_set.dropped}); "
          f"val {len(val_set)} utts (dropped {val_set.dropped})", flush=True)

    sampler = LengthBatchSampler(train_set.samples, settings["max_samples_per_batch"], settings["seed"])
    val_sampler = LengthBatchSampler(
        val_set.samples, settings["max_samples_per_eval_batch"], settings["seed"], shuffle=False
    )
    workers = settings["num_workers"]
    train_loader = make_loader(train_set, sampler, extractor, workers)
    val_loader = make_loader(val_set, val_sampler, extractor, workers)

    model, head = build_model(config["model"], vocab)
    model.to(device)
    print(f"head: {head}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=settings["learning_rate"],
        betas=tuple(settings["adam_betas"]),
        weight_decay=settings["weight_decay"],
    )
    total = settings["max_updates"]
    scheduler = LambdaLR(
        optimizer,
        lambda u: tri_stage(u, total, settings["warmup_fraction"], settings["hold_fraction"]),
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=GRAD_SCALE_INIT, enabled=use_amp)

    run_dir = data_path(settings["output_dir"]) / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    position = {"update": 0, "epoch": 0, "batch": 0, "best_wer": float("inf"), "skipped": 0}

    source = latest_checkpoint(run_dir) if resume == "auto" else Path(resume) if resume else None
    if source is not None:
        from safetensors.torch import load_file

        model.load_state_dict(load_file(source / "model.safetensors"))
        state = torch.load(source / "trainer_state.pt", map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        position = state["position"]
        restore_rng(state["rng"])
        print(f"resumed from {source} at update {position['update']}", flush=True)

    (run_dir / "run_meta.json").write_text(
        json.dumps({"config": config, "meta": run_metadata(config)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log_path = run_dir / "train_log.csv"
    new_log = not log_path.exists()
    log_file = log_path.open("a", newline="", encoding="utf-8")
    log = csv.DictWriter(log_file, LOG_FIELDS)
    if new_log:
        log.writeheader()

    accumulate = settings["gradient_accumulation"]
    frozen_until = settings["freeze_encoder_updates"]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    window_loss, window_batches, last_norm, started = 0.0, 0, 0.0, time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    update, epoch, batch_index = position["update"], position["epoch"], position["batch"]
    final_metrics: dict[str, float] = {}
    while update < total:
        sampler.set_position(epoch, batch_index)
        for batch in train_loader:
            set_encoder_frozen(model, update < frozen_until)
            inputs = to_device(batch, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**inputs).loss / accumulate
            batch_index += 1
            if not torch.isfinite(loss):
                position["skipped"] += 1
                continue
            scaler.scale(loss).backward()
            window_loss += loss.item() * accumulate
            window_batches += 1
            if batch_index % accumulate:
                continue

            scaler.unscale_(optimizer)
            last_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"]).item()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1

            validate = update % settings["eval_every"] == 0 or update == total
            if update % settings["log_every"] == 0 or validate:
                row = {
                    "update": update,
                    "epoch": epoch,
                    "train_loss": round(window_loss / max(window_batches, 1), 4),
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": round(last_norm, 3),
                    "seconds": round(time.time() - started, 1),
                    "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20)
                    if device.type == "cuda" else 0,
                    "skipped_batches": position["skipped"],
                }
                window_loss, window_batches = 0.0, 0
                if validate:
                    final_metrics = evaluate(model, val_loader, vocab, device, use_amp)
                    row.update({k: round(v, 4) for k, v in final_metrics.items()})
                print(row, flush=True)
                log.writerow(row)
                log_file.flush()

            if validate:
                position.update(update=update, epoch=epoch, batch=batch_index)
                directory = run_dir / f"step-{update:06d}"
                save_checkpoint(directory, model, vocab, extractor, optimizer, scheduler, scaler, position)
                if final_metrics["val_wer"] < position["best_wer"]:
                    position["best_wer"] = final_metrics["val_wer"]
                    save_model(run_dir / "best", model, vocab, extractor)
                    save_checkpoint(directory, model, vocab, extractor, optimizer, scheduler, scaler, position)
                prune_checkpoints(run_dir, settings["keep_checkpoints"])
            if update >= total:
                break
        else:
            # A partial accumulation window does not carry over to a new epoch.
            optimizer.zero_grad(set_to_none=True)
            epoch, batch_index = epoch + 1, 0
    log_file.close()
    return {
        "updates": update,
        "epochs_started": epoch + 1,
        "best_wer": position["best_wer"],
        "last": final_metrics,
        "run_dir": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="'auto' for the latest checkpoint, or a checkpoint path")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    print(train(load_config(args.config, args.set), args.resume))


if __name__ == "__main__":
    main()
