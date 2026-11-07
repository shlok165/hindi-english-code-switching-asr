"""wav2vec 2.0 encoder with a linear CTC head sized to our vocabulary.

How CTC handles unaligned audio and text: the encoder emits one distribution
over vocabulary + blank about every 20 ms. CTC needs no frame labels. It sums
the probability of every frame-level path that reduces to the transcript after
merging repeats and removing blanks, so a transcript only has to be no longer
than the frame sequence. A blank between two identical characters is what lets
a real double letter survive the merge.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from src.models.tokenizer import BLANK, BOS, EOS, UNK, ctc_collapse, decode_labels

NEVER_TARGETS = (BOS, EOS, UNK)


def retarget_head(
    old_head: nn.Linear,
    source_vocab: dict[str, int],
    vocab: dict[str, int],
    source_blank: str,
    init_std: float,
) -> tuple[nn.Linear, list[str]]:
    """Build a head for vocab, copying the rows the checkpoint already has.

    Our blank is `<pad>`, but the row that holds the checkpoint's blank must
    be named explicitly. Checkpoints exported from fairseq use `<s>` (id 0) as
    the CTC blank while their config says pad_token_id=1, so trusting the config
    would give training the wrong blank. Other specials are never targets, so
    they get a fresh initialisation and cannot inherit a blank-like row.
    """
    new_head = nn.Linear(old_head.in_features, len(vocab))
    nn.init.normal_(new_head.weight, std=init_std)
    nn.init.zeros_(new_head.bias)
    copied = []
    with torch.no_grad():
        for token, index in vocab.items():
            if token in NEVER_TARGETS:
                continue
            source = source_vocab.get(source_blank if token == BLANK else token)
            if source is not None and source < old_head.out_features:
                new_head.weight[index] = old_head.weight[source]
                new_head.bias[index] = old_head.bias[source]
                copied.append(token)
    return new_head, copied


def build_model(model_config: dict[str, Any], vocab: dict[str, int]) -> tuple[Wav2Vec2ForCTC, dict]:
    """Load the pretrained model and retarget its head at vocab.

    Rows for characters the checkpoint already knows, including blank and the
    word delimiter, keep their weights. New characters, such as Latin letters
    on a Devanagari checkpoint, start from a small random initialisation.
    """
    name = model_config["pretrained"]
    model, loading = Wav2Vec2ForCTC.from_pretrained(
        name,
        output_loading_info=True,
        ctc_loss_reduction=model_config["ctc_loss_reduction"],
        ctc_zero_infinity=model_config["ctc_zero_infinity"],
        mask_time_prob=model_config["mask_time_prob"],
        mask_time_length=model_config["mask_time_length"],
    )
    if loading["missing_keys"]:
        raise RuntimeError(f"{name}: checkpoint is missing {sorted(loading['missing_keys'])[:5]}")

    source_vocab = Wav2Vec2CTCTokenizer.from_pretrained(name).get_vocab()
    new_head, copied = retarget_head(
        model.lm_head, source_vocab, vocab, model_config["source_blank"], model.config.initializer_range
    )
    model.lm_head = new_head
    model.config.vocab_size = len(vocab)
    model.config.pad_token_id = vocab["<pad>"]
    model.config.bos_token_id = vocab["<s>"]
    model.config.eos_token_id = vocab["</s>"]

    if model_config.get("freeze_feature_encoder", True):
        model.freeze_feature_encoder()
    if model_config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, {
        "head_rows_copied": len(copied),
        "head_rows_new": len(vocab) - len(copied),
        "new_tokens": sorted(set(vocab) - set(copied)),
    }


def set_encoder_frozen(model: Wav2Vec2ForCTC, frozen: bool) -> None:
    """Freeze or release the transformer. The CNN is controlled separately."""
    for block in (model.wav2vec2.encoder, model.wav2vec2.feature_projection):
        for parameter in block.parameters():
            parameter.requires_grad = not frozen


def output_lengths(model: Wav2Vec2ForCTC, sample_lengths: torch.Tensor) -> torch.Tensor:
    """Number of CTC frames the CNN produces for each raw-audio length."""
    return model._get_feat_extract_output_lengths(sample_lengths).long()


def greedy_decode(
    logits: torch.Tensor, lengths: torch.Tensor, vocab: dict[str, int]
) -> list[str]:
    """Best-path decode, truncated to each utterance's true length.

    Frames beyond the true length come from padding. They must not reach the
    collapse step, or padding could add spurious characters.
    """
    blank = vocab[BLANK]
    hidden = {vocab[token] for token in NEVER_TARGETS}
    best = logits.argmax(dim=-1).cpu()
    decoded = []
    for row, length in zip(best, lengths.tolist()):
        ids = [i for i in ctc_collapse(row[:length].tolist(), blank) if i not in hidden]
        decoded.append(" ".join(decode_labels(ids, vocab).split()))
    return decoded


def save_model(
    directory: Path,
    model: Wav2Vec2ForCTC,
    vocab: dict[str, int],
    feature_extractor: Wav2Vec2FeatureExtractor,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    feature_extractor.save_pretrained(directory)
    text = json.dumps(vocab, ensure_ascii=False, indent=0)
    (directory / "vocab.json").write_text(text, encoding="utf-8")


def load_model(directory: str | Path) -> tuple[Wav2Vec2ForCTC, dict[str, int], Wav2Vec2FeatureExtractor]:
    directory = Path(directory)
    vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
    return (
        Wav2Vec2ForCTC.from_pretrained(directory),
        vocab,
        Wav2Vec2FeatureExtractor.from_pretrained(directory),
    )


def replace_directory(temporary: Path, final: Path) -> None:
    """Swap a finished checkpoint into place so a crash cannot leave half of one."""
    if final.exists():
        shutil.rmtree(final)
    temporary.replace(final)
