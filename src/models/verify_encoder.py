"""Check that a pretrained checkpoint really fits this pipeline, before training.

    python -m src.models.verify_encoder --config configs/model_him4200.yaml

Writes results/encoder_<name>.json. Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys

from collections import Counter

import numpy as np
import torch
from transformers import AutoConfig, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from src import ROOT
from src.data.dataset import read_manifest
from src.data.preprocessing import TARGET_SAMPLE_RATE, load_audio
from src.models.ctc_model import build_model, output_lengths
from src.models.tokenizer import ctc_collapse
from src.utils.config import data_path, load_config


def measure_blank(model, tokenizer, extractor, configured: str, count: int = 8) -> dict | None:
    """Find the real CTC blank: on speech it is the most frequent frame label.

    A config's pad_token_id cannot be trusted for this; see retarget_head. Uses
    the validation split, and returns None before manifests exist.
    """
    manifest = data_path("data/processed/manifests") / "val.jsonl"
    if not manifest.exists():
        return None
    entries = read_manifest(manifest)[:: 200][:count]
    inverse = {index: token for token, index in tokenizer.get_vocab().items()}
    frames: Counter[str] = Counter()
    outputs = []
    for entry in entries:
        audio = load_audio(str(data_path(entry["audio"])), entry["start"], entry["end"])
        values = extractor(audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt").input_values
        with torch.no_grad():
            best = model(values).logits[0].argmax(-1).tolist()
        tokens = [inverse[i] for i in best]
        frames.update(tokens)
    dominant, top = frames.most_common(1)[0]
    blank_id = tokenizer.get_vocab()[dominant]
    text = "".join(inverse[i] for i in ctc_collapse([tokenizer.get_vocab()[t] for t in tokens], blank_id))
    return {
        "dominant_token": dominant,
        "dominant_frame_percent": round(100 * top / sum(frames.values()), 1),
        "config_pad_token": tokenizer.pad_token,
        "configured_source_blank": configured,
        "configured_blank_is_correct": dominant == configured,
        "sample_reference": entries[-1]["text"],
        "sample_pretrained_output": text.replace("|", " "),
    }


def verify(model_config: dict, vocab: dict[str, int] | None) -> dict:
    name = model_config["pretrained"]
    report: dict = {"checkpoint": name}
    try:
        config = AutoConfig.from_pretrained(name)
    except OSError as error:
        report.update(accessible=False, error=str(error).splitlines()[0],
                      action="Accept the terms on the model page, then run: hf auth login")
        return report
    report["accessible"] = True

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(name)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(name)
    report["architecture"] = {
        "class": config.architectures,
        "hidden_size": config.hidden_size,
        "layers": config.num_hidden_layers,
        "heads": config.num_attention_heads,
        "feat_extract_norm": config.feat_extract_norm,
        "do_stable_layer_norm": config.do_stable_layer_norm,
        "downsampling": int(np.prod(config.conv_stride)),
    }
    report["feature_extractor"] = {
        "sampling_rate": extractor.sampling_rate,
        "do_normalize": extractor.do_normalize,
        "return_attention_mask": extractor.return_attention_mask,
        "matches_16k_data": extractor.sampling_rate == TARGET_SAMPLE_RATE,
    }
    source_vocab = tokenizer.get_vocab()
    report["tokenizer"] = {
        "size": len(source_vocab),
        "blank": tokenizer.pad_token,
        "blank_id": tokenizer.pad_token_id,
        "delimiter": tokenizer.word_delimiter_token,
        "has_latin": any("a" <= token <= "z" for token in source_vocab if len(token) == 1),
    }

    model = Wav2Vec2ForCTC.from_pretrained(name).eval()
    report["parameters_million"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    samples = torch.zeros(1, TARGET_SAMPLE_RATE * 2)
    samples[0] = torch.randn(TARGET_SAMPLE_RATE * 2) * 0.1
    with torch.no_grad():
        output = model(samples, output_hidden_states=True)
    frames = int(output_lengths(model, torch.tensor([samples.shape[1]]))[0])
    report["forward"] = {
        "input_samples": samples.shape[1],
        "last_hidden_shape": list(output.hidden_states[-1].shape),
        "logits_shape": list(output.logits.shape),
        "frames_match_length_formula": frames == output.logits.shape[1],
        "hidden_size_matches_head": output.hidden_states[-1].shape[-1] == model.lm_head.in_features,
    }

    report["blank"] = measure_blank(model, tokenizer, extractor, model_config["source_blank"])

    if vocab is not None:
        retargeted, head = build_model(model_config, vocab)
        retargeted.eval()
        with torch.no_grad():
            logits = retargeted(samples).logits
        report["retargeted_head"] = {
            **head,
            "new_tokens": head["new_tokens"],
            "logits_shape": list(logits.shape),
            "vocab_size_matches": logits.shape[-1] == len(vocab),
        }

    checks = [
        report["feature_extractor"]["matches_16k_data"],
        report["forward"]["frames_match_length_formula"],
        report["forward"]["hidden_size_matches_head"],
        vocab is None or report["retargeted_head"]["vocab_size_matches"],
        report["blank"] is None or report["blank"]["configured_blank_is_correct"],
    ]
    report["compatible"] = all(checks)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args()
    vocab_path = data_path(load_config(args.data_config)["data"]["tokenizer_dir"]) / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8")) if vocab_path.exists() else None

    report = verify(load_config(args.config)["model"], vocab)
    slug = report["checkpoint"].split("/")[-1]
    output = ROOT / "results" / f"encoder_{slug}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if report.get("compatible") else 1)


if __name__ == "__main__":
    main()
