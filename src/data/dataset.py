"""PyTorch dataset, length-aware batching and padding for CTC training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from transformers import Wav2Vec2FeatureExtractor

from src.data.preprocessing import TARGET_SAMPLE_RATE, load_audio
from src.models.tokenizer import encode
from src.utils.config import data_path, read_manifest

# Labels must fit in the CTC output. wav2vec 2.0 downsamples by 320, so this
# many samples yield one frame, ignoring the small loss at the utterance edges.
SAMPLES_PER_FRAME = 320
PAD_LABEL = -100
TRAINABLE_SPLITS = {"train", "val"}


class MucsDataset(Dataset):
    """Utterances cut on demand from the long tutorial recordings."""

    def __init__(
        self,
        manifest: Path,
        vocab: dict[str, int],
        min_seconds: float = 0.0,
        max_seconds: float = float("inf"),
        limit: int | None = None,
    ) -> None:
        self.vocab = vocab
        entries = read_manifest(manifest)
        kept, self.dropped = [], {"empty_labels": 0, "duration": 0, "labels_exceed_frames": 0}
        for entry in entries:
            labels = encode(entry["text"], vocab)
            samples = round(entry["duration"] * TARGET_SAMPLE_RATE)
            if not labels:
                self.dropped["empty_labels"] += 1
            elif not min_seconds <= entry["duration"] <= max_seconds:
                self.dropped["duration"] += 1
            elif len(labels) > samples // SAMPLES_PER_FRAME:
                self.dropped["labels_exceed_frames"] += 1
            else:
                kept.append((entry, labels, samples))
        kept = kept[:limit] if limit else kept
        self.entries = [item[0] for item in kept]
        self.labels = [item[1] for item in kept]
        self.samples = np.array([item[2] for item in kept], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        return {
            "id": entry["id"],
            "text": entry["text"],
            "audio": load_audio(str(data_path(entry["audio"])), entry["start"], entry["end"]),
            "labels": self.labels[index],
        }


@dataclass
class Collator:
    """Normalises and pads audio, and pads labels with -100, which CTC ignores."""

    feature_extractor: Wav2Vec2FeatureExtractor

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        features = self.feature_extractor(
            [item["audio"] for item in items],
            sampling_rate=TARGET_SAMPLE_RATE,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        longest = max(len(item["labels"]) for item in items)
        labels = torch.full((len(items), longest), PAD_LABEL, dtype=torch.long)
        for row, item in enumerate(items):
            labels[row, : len(item["labels"])] = torch.tensor(item["labels"], dtype=torch.long)
        return {
            "input_values": features["input_values"],
            "attention_mask": features["attention_mask"].long(),
            "labels": labels,
            "ids": [item["id"] for item in items],
            "texts": [item["text"] for item in items],
        }


class LengthBatchSampler(Sampler[list[int]]):
    """Batches of similar length, capped by padded samples rather than count.

    GPU memory follows longest utterance x batch size, so a fixed batch size
    would run out of memory on long utterances and waste it on short ones.
    Batch order depends only on (seed, epoch), so resuming needs just the epoch
    and the position within it, and no skipped audio has to be read.
    """

    CHUNK_BATCHES = 50

    def __init__(self, lengths: np.ndarray, max_samples: int, seed: int, shuffle: bool = True,
                 max_utterances: int | None = None) -> None:
        if lengths.max(initial=0) > max_samples:
            raise ValueError(f"an utterance has {lengths.max()} samples, over max_samples={max_samples}")
        self.lengths = lengths
        self.max_samples = max_samples
        self.max_utterances = max_utterances or len(lengths) or 1
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.start = 0

    def batches(self, epoch: int) -> list[list[int]]:
        rng = np.random.default_rng([self.seed, epoch])
        count = len(self.lengths)
        if self.shuffle:
            order = rng.permutation(count)
            typical = max(int(self.max_samples / max(self.lengths.mean(), 1)), 1)
            typical = min(typical, self.max_utterances)
            size = typical * self.CHUNK_BATCHES
            chunks = [order[i : i + size] for i in range(0, count, size)]
            order = np.concatenate([c[np.argsort(-self.lengths[c], kind="stable")] for c in chunks])
        else:
            order = np.argsort(-self.lengths, kind="stable")

        batches, current, longest = [], [], 0
        for index in order.tolist():
            candidate = max(longest, int(self.lengths[index]))
            full = len(current) >= self.max_utterances
            if current and (full or candidate * (len(current) + 1) > self.max_samples):
                batches.append(current)
                current, candidate = [], int(self.lengths[index])
            current.append(index)
            longest = candidate
        if current:
            batches.append(current)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def set_position(self, epoch: int, start: int = 0) -> None:
        self.epoch, self.start = epoch, start

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.batches(self.epoch)[self.start :]

    def __len__(self) -> int:
        return len(self.batches(self.epoch)) - self.start
