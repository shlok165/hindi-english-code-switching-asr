"""End-to-end smoke test on real audio. Every step raises if it fails.

    python -m src.training.smoke_test

Covers: load, preprocess, encode, batch, load model, forward, finite CTC loss,
one training step, save, reload and identical inference. It then overfits a few
utterances to show the pipeline can learn, and proves that resuming from a
checkpoint matches an uninterrupted run. Writes results/smoke_test.json.

None of this measures accuracy, and it is not full-scale training.
"""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Wav2Vec2FeatureExtractor

from src import ROOT
from src.data.dataset import Collator, LengthBatchSampler
from src.data.preprocessing import TARGET_SAMPLE_RATE
from src.metrics.wer import score
from src.models.ctc_model import build_model, greedy_decode, load_model, output_lengths, save_model
from src.models.tokenizer import decode_labels, encode
from src.training.train import GRAD_SCALE_INIT, build_dataset, to_device, train
from src.utils.config import data_path, load_config, run_metadata, seed_everything

CONFIG = "configs/train_smoke.yaml"
BATCH_SIZE = 4
OVERFIT_STEPS = 500
MAX_OVERFIT_WER = 30.0  # A falling loss alone is not proof: all-blank output lowers it too


def step(results: dict, name: str, **facts) -> None:
    results["steps"][name] = facts
    print(f"PASS {name}: {facts}", flush=True)


def infer(model, batch, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        logits = model(batch["input_values"].to(device),
                       attention_mask=batch["attention_mask"].to(device)).logits
    lengths = output_lengths(model, batch["attention_mask"].sum(-1).to(device))
    return logits, lengths


def resume_matches(config: dict, directory: Path) -> dict:
    """Compare 6 straight updates with 3 + resume + 3, against a noise floor.

    GPU floating point is not perfectly repeatable, so the same run twice sets
    the floor. Resuming must be as close as that repeat is, within a small margin.
    """
    config = copy.deepcopy(config)
    config["training"]["output_dir"] = str(directory)
    config["training"]["keep_checkpoints"] = 10

    def final_weights(run: str) -> dict:
        return load_file(directory / run / "step-000006" / "model.safetensors")

    def largest_difference(first: dict, second: dict) -> float:
        return max(float((first[k].float() - second[k].float()).abs().max()) for k in first)

    for run in ("straight", "repeat"):
        straight = copy.deepcopy(config)
        straight["run_name"] = run
        train(straight)
    noise = largest_difference(final_weights("straight"), final_weights("repeat"))

    # eval_every=3 leaves a checkpoint at update 3. Copying it into a new run
    # means the second half starts from disk alone, as after a real crash.
    resumed_dir = directory / "resumed"
    resumed_dir.mkdir()
    shutil.copytree(directory / "straight" / "step-000003", resumed_dir / "step-000003")
    resumed = copy.deepcopy(config)
    resumed["run_name"] = "resumed"
    train(resumed, resume="auto")

    difference = largest_difference(final_weights("straight"), final_weights("resumed"))
    if difference > max(5 * noise, 1e-4):
        raise AssertionError(f"resume drifted by {difference}; same-run noise is {noise}")
    return {"repeat_run_difference": noise, "resume_difference": difference}


def main() -> None:
    config = load_config(CONFIG)
    seed_everything(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict = {"device": str(device), "meta": run_metadata(config), "steps": {}}

    vocab_path = data_path(config["data"]["tokenizer_dir"]) / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    dataset = build_dataset(config, "train", vocab)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(config["model"]["pretrained"])

    items = [dataset[i] for i in range(BATCH_SIZE)]
    audio = items[0]["audio"]
    if audio.ndim != 1 or audio.dtype != np.float32 or not np.isfinite(audio).all():
        raise AssertionError("audio is not finite float32 mono")
    step(results, "1_load_audio", utterances=len(items),
         seconds=[round(len(i["audio"]) / TARGET_SAMPLE_RATE, 2) for i in items],
         peak=round(float(np.abs(audio).max()), 3))

    step(results, "2_preprocess", id=items[0]["id"], text=items[0]["text"])

    for item in items:
        if decode_labels(item["labels"], vocab) != " ".join(
            "".join(c for c in item["text"] if c == " " or c in vocab).split()
        ):
            raise AssertionError("label round trip failed")
    step(results, "3_encode", label_lengths=[len(i["labels"]) for i in items], round_trip=True)

    collate = Collator(extractor)
    batch = collate(items)
    frames_ok = batch["attention_mask"].sum(-1).tolist() == [len(i["audio"]) for i in items]
    if not frames_ok:
        raise AssertionError("attention mask disagrees with audio lengths")
    step(results, "4_batch", input_shape=list(batch["input_values"].shape),
         label_shape=list(batch["labels"].shape), mask_matches_lengths=frames_ok)

    model, head = build_model(config["model"], vocab)
    model.to(device).train()
    step(results, "5_load_model", **head)

    inputs = to_device(batch, device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        output = model(**inputs)
    frames = output_lengths(model, inputs["attention_mask"].sum(-1))
    longest_label = (inputs["labels"] != -100).sum(-1)
    if not (frames >= longest_label).all() or output.logits.shape[-1] != len(vocab):
        raise AssertionError("labels do not fit the CTC output")
    step(results, "6_forward", logits_shape=list(output.logits.shape),
         output_frames=frames.tolist(), label_lengths=longest_label.tolist())

    if not torch.isfinite(output.loss):
        raise AssertionError(f"CTC loss is {output.loss}")
    step(results, "7_finite_ctc_loss", loss=round(output.loss.item(), 4))

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=GRAD_SCALE_INIT, enabled=device.type == "cuda")
    watched = model.lm_head.weight.detach().clone()
    frozen = next(model.wav2vec2.feature_extractor.parameters()).detach().clone()
    scaler.scale(output.loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    moved = float((model.lm_head.weight - watched).abs().max())
    cnn_moved = not torch.equal(frozen, next(model.wav2vec2.feature_extractor.parameters()))
    cnn_frozen = config["model"].get("freeze_feature_encoder", True)
    if moved == 0 or cnn_moved == cnn_frozen:
        raise AssertionError(f"head moved {moved}; CNN moved {cnn_moved}, frozen {cnn_frozen}")
    step(results, "8_training_step", head_weight_change=moved, cnn_frozen=cnn_frozen, cnn_moved=cnn_moved)

    temporary = Path(tempfile.mkdtemp(prefix="smoke_", dir=data_path("data")))
    try:
        directory = temporary / "checkpoint"
        save_model(directory, model, vocab, extractor)
        size = sum(f.stat().st_size for f in directory.iterdir())
        step(results, "9_save_checkpoint", files=sorted(f.name for f in directory.iterdir()),
             megabytes=round(size / 1e6, 1))

        before, lengths = infer(model, batch, device)
        reloaded, reloaded_vocab, _ = load_model(directory)
        reloaded.to(device)
        after, _ = infer(reloaded, batch, device)
        difference = float((before - after).abs().max())
        if reloaded_vocab != vocab or difference > 1e-5:
            raise AssertionError(f"reload changed outputs by {difference}")
        step(results, "10_reload_and_infer", largest_logit_difference=difference,
             sample_output=greedy_decode(after, lengths, vocab)[0][:80])
        del reloaded

        # A pipeline that cannot memorise four utterances cannot train at all.
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", init_scale=GRAD_SCALE_INIT, enabled=device.type == "cuda")
        losses = []
        for _ in range(OVERFIT_STEPS):
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(**inputs).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
        logits, lengths = infer(model, batch, device)
        decoded = greedy_decode(logits, lengths, vocab)
        references = [" ".join("".join(c for c in i["text"] if c == " " or c in vocab).split()) for i in items]
        pretrained_logits = before
        wer_before = score(zip(references, greedy_decode(pretrained_logits, lengths, vocab))).wer
        wer_after = score(zip(references, decoded)).wer
        if wer_after > MAX_OVERFIT_WER or wer_after >= wer_before:
            raise AssertionError(f"failed to memorise the batch: WER {wer_before} -> {wer_after}")
        step(results, "11_overfit_small_batch", steps=OVERFIT_STEPS,
             first_loss=round(losses[0], 4), last_loss=round(losses[-1], 4),
             wer_before=round(wer_before, 2), wer_after=round(wer_after, 2),
             reference=references[0][:80], output=decoded[0][:80])
        del model, optimizer, scaler
        torch.cuda.empty_cache()

        comparison = resume_matches(config, temporary / "resume")
        step(results, "12_resume_matches_straight_run", **comparison)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    output = ROOT / "results" / "smoke_test.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ALL SMOKE STEPS PASSED")


if __name__ == "__main__":
    main()
