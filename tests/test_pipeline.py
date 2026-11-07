"""Fast unit tests. No dataset, network or GPU needed."""

import json

import numpy as np
import pytest
import soundfile as sf
import torch

import src  # noqa: F401  (sets the cache location before transformers loads)
from src.data.dataset import PAD_LABEL, Collator, LengthBatchSampler, MucsDataset
from src.data.manifest import split_validation, summarise
from src.data.preprocessing import clean_transcript, load_audio, normalise_unicode, resample, to_mono
from src.metrics.wer import align, count, score
from src.models.tokenizer import (
    build_vocab,
    ctc_collapse,
    decode_labels,
    encode,
    make_tokenizer,
    save_tokenizer,
)
from src.training.train import manifest_path, tri_stage
from src.utils.config import apply_overrides, deep_merge


def test_transcript_keeps_both_scripts():
    assert clean_transcript("Type करें: gcc   Space") == "type करें gcc space"


def test_transcript_restores_visarga_but_strips_punctuation_colon():
    assert clean_transcript("अत: आगे बढ़ें:") == "अतः आगे बढ़ें"


def test_transcript_keeps_spoken_symbols():
    assert clean_transcript("ctrl + alt / a * b, c.") == "ctrl + alt / a * b c"


def test_nukta_spellings_become_identical():
    precomposed, decomposed = "ज़", "ज़"
    assert normalise_unicode(precomposed) == normalise_unicode(decomposed)


def test_zero_width_characters_removed():
    assert normalise_unicode("क‍ख") == "कख"


def test_no_transliteration_happens():
    assert clean_transcript("file फाइल") == "file फाइल"


def test_mono_and_resample():
    assert to_mono(np.ones((10, 2), dtype=np.float32)).shape == (10,)
    silence = np.zeros(8000, dtype=np.float32)
    assert resample(silence, 16000) is silence


def test_load_audio_cuts_the_requested_range(tmp_path):
    path = tmp_path / "a.wav"
    ramp = (np.arange(16000) / 16000).astype(np.float32)
    sf.write(path, ramp, 16000, subtype="FLOAT")
    cut = load_audio(str(path), 0.25, 0.5)
    assert len(cut) == 4000
    assert cut[0] == pytest.approx(0.25, abs=1e-6)


def test_load_audio_rejects_empty_range(tmp_path):
    path = tmp_path / "a.wav"
    sf.write(path, np.zeros(100, dtype=np.float32), 16000)
    with pytest.raises(ValueError):
        load_audio(str(path), 5.0, 6.0)


def test_vocab_has_fixed_specials_and_applies_cutoff():
    vocab, dropped = build_vocab(["aab"] * 5 + ["z"], min_count=5)
    assert vocab["<pad>"] == 0 and vocab["|"] == 4
    assert "a" in vocab and "b" in vocab and "z" not in vocab
    assert dropped == {"z": 1}


def test_vocab_rejects_reserved_delimiter():
    with pytest.raises(ValueError):
        build_vocab(["a|b"] * 20)


def test_round_trip_mixed_script():
    text = "file को save करें 2"
    vocab, _ = build_vocab([text] * 20)
    assert decode_labels(encode(text, vocab), vocab) == text


def test_repeated_letters_survive_label_decoding():
    vocab, _ = build_vocab(["aa"] * 20)
    assert decode_labels(encode("aa", vocab), vocab) == "aa"


def test_encode_drops_unknown_characters():
    vocab, _ = build_vocab(["ab"] * 20)
    assert decode_labels(encode("a z b", vocab), vocab) == "a b"


def test_hf_tokenizer_agrees_with_encode(tmp_path):
    text = "file को save करें"
    vocab, _ = build_vocab([text] * 20)
    tokenizer = save_tokenizer(vocab, tmp_path)
    assert tokenizer(text).input_ids == encode(text, vocab)
    assert tokenizer.pad_token_id == vocab["<pad>"]
    assert make_tokenizer(tmp_path / "vocab.json").get_vocab() == tokenizer.get_vocab()


def test_ctc_collapse_merges_repeats_then_removes_blanks():
    assert ctc_collapse([0, 5, 5, 0, 5, 6, 6, 0], blank=0) == [5, 5, 6]


def test_sampler_respects_cap_and_covers_everything():
    lengths = np.random.default_rng(0).integers(8000, 160000, 500)
    sampler = LengthBatchSampler(lengths, 320000, seed=1)
    batches = sampler.batches(0)
    assert sorted(i for b in batches for i in b) == list(range(500))
    assert all(max(lengths[b]) * len(b) <= 320000 for b in batches)


def test_sampler_is_deterministic_per_epoch_and_varies_across_epochs():
    lengths = np.random.default_rng(0).integers(8000, 160000, 500)
    sampler = LengthBatchSampler(lengths, 320000, seed=1)
    assert sampler.batches(3) == sampler.batches(3)
    assert sampler.batches(3) != sampler.batches(4)


def test_sampler_resume_position_skips_consumed_batches():
    lengths = np.random.default_rng(0).integers(8000, 160000, 100)
    sampler = LengthBatchSampler(lengths, 320000, seed=1)
    everything = sampler.batches(2)
    sampler.set_position(2, 3)
    assert list(sampler) == everything[3:]


def test_sampler_rejects_oversized_utterance():
    with pytest.raises(ValueError):
        LengthBatchSampler(np.array([400000]), 320000, seed=1)


def test_collator_pads_audio_and_labels():
    from transformers import Wav2Vec2FeatureExtractor

    extractor = Wav2Vec2FeatureExtractor(sampling_rate=16000, do_normalize=True,
                                         return_attention_mask=True)
    rng = np.random.default_rng(0)
    items = [
        {"id": "a", "text": "x", "audio": rng.random(1600, dtype=np.float32), "labels": [5, 6, 7]},
        {"id": "b", "text": "y", "audio": rng.random(800, dtype=np.float32), "labels": [5]},
    ]
    batch = Collator(extractor)(items)
    assert batch["input_values"].shape == (2, 1600)
    assert batch["attention_mask"].sum(-1).tolist() == [1600, 800]
    assert batch["labels"].tolist() == [[5, 6, 7], [5, PAD_LABEL, PAD_LABEL]]


def test_dataset_filters_bad_entries(tmp_path):
    entries = [
        {"id": "ok", "audio": "a.wav", "start": 0, "end": 2, "duration": 2.0, "text": "ab"},
        {"id": "empty", "audio": "a.wav", "start": 0, "end": 2, "duration": 2.0, "text": ""},
        {"id": "long", "audio": "a.wav", "start": 0, "end": 30, "duration": 30.0, "text": "ab"},
        {"id": "dense", "audio": "a.wav", "start": 0, "end": 0.6, "duration": 0.6, "text": "ab" * 40},
    ]
    path = tmp_path / "m.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
    vocab, _ = build_vocab(["ab"] * 20)
    dataset = MucsDataset(path, vocab, 0.5, 20.0)
    assert [e["id"] for e in dataset.entries] == ["ok"]
    assert dataset.dropped == {"empty_labels": 1, "duration": 1, "labels_exceed_frames": 1}


@pytest.mark.parametrize("split", ["test", "blind"])
def test_training_refuses_evaluation_splits(split):
    with pytest.raises(ValueError):
        manifest_path({"data": {"manifest_dir": "data/processed/manifests"}}, split)


def test_validation_split_holds_out_whole_recordings():
    entries = [{"id": f"{r}_{i}", "recording": r, "duration": 600.0, "split": "train"}
               for r in "abcdefgh" for i in range(6)]
    train, val = split_validation(entries, hours=2.0, seed=1)
    assert not {e["recording"] for e in train} & {e["recording"] for e in val}
    assert sum(e["duration"] for e in val) >= 7200
    assert all(e["split"] == "val" for e in val)
    assert split_validation(entries, 2.0, 1)[1] == val


def test_manifest_paths_resolve_under_a_custom_data_root(tmp_path, monkeypatch):
    from src.data.manifest import portable_path
    from src.utils.config import data_path

    extracted = tmp_path / "interim" / "extracted"
    wav = extracted / "test" / "a.wav"
    assert portable_path(wav, extracted) == "data/interim/extracted/test/a.wav"
    monkeypatch.setenv("MUCS_DATA_ROOT", str(tmp_path))
    assert data_path(portable_path(wav, extracted)) == wav


def test_summary_counts_scripts_and_overlap():
    entries = [{"text": "a क", "duration": 1.0, "recording": "r"},
               {"text": "", "duration": 1.0, "recording": "r"}]
    summary = summarise(entries, {"a क"})
    assert summary["latin_word_percent"] == 50 and summary["devanagari_word_percent"] == 50
    assert summary["empty_transcripts"] == 1
    assert summary["utterances_with_sentence_in_train_percent"] == 100


def test_tri_stage_shape():
    values = [tri_stage(u, 100, 0.1, 0.4) for u in range(100)]
    assert values[0] == pytest.approx(0.1)
    assert values[9] == 1.0 and values[49] == 1.0
    assert values[99] < 0.01
    assert all(a >= b for a, b in zip(values[50:], values[51:]))


def test_wer_counts_each_error_type():
    result = count("a b c d".split(), "a x c e f".split())
    assert (result.substitutions, result.deletions, result.insertions) == (2, 0, 1)
    assert count("a b c".split(), "a c".split()).deletions == 1
    assert count("a b".split(), "a b c".split()).insertions == 1


def test_wer_agrees_with_jiwer_on_random_sentences():
    jiwer = pytest.importorskip("jiwer")
    rng = np.random.default_rng(0)
    words = list("abcdefgh")
    pairs = [(" ".join(rng.choice(words, rng.integers(1, 15))),
              " ".join(rng.choice(words, rng.integers(1, 15)))) for _ in range(200)]
    expected = sum(jiwer.process_words(r, h).substitutions + jiwer.process_words(r, h).deletions
                   + jiwer.process_words(r, h).insertions for r, h in pairs)
    assert score(pairs).counts.errors == expected


def test_alignment_replays_to_hypothesis():
    ref, hyp = "a b c d".split(), "b c x".split()
    rebuilt = [hyp[j] for op, _, j in align(ref, hyp) if j is not None]
    assert rebuilt == hyp


def test_config_merge_and_overrides():
    merged = deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}})
    assert merged == {"a": {"x": 1, "y": 3}}
    assert apply_overrides(merged, ["a.x=1e-4", "b.c=true"]) == {"a": {"x": 1e-4, "y": 3}, "b": {"c": True}}


def test_head_copies_the_named_blank_row_not_the_pad_row():
    from torch import nn

    from src.models.ctc_model import retarget_head

    # Fairseq layout: <s> is the real blank, and <pad> is an untrained row.
    source_vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "|": 4, "क": 5}
    old = nn.Linear(2, 6)
    with torch.no_grad():
        old.weight.copy_(torch.arange(12.0).reshape(6, 2))
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "a": 5, "क": 6}
    new, copied = retarget_head(old, source_vocab, vocab, "<s>", 0.02)
    assert new.weight[0].tolist() == old.weight[0].tolist()
    assert new.weight[6].tolist() == old.weight[5].tolist()
    assert new.weight[1].tolist() != old.weight[0].tolist()
    assert sorted(copied) == sorted(["<pad>", "|", "क"])


def test_greedy_decode_hides_special_tokens():
    from src.models.ctc_model import greedy_decode

    vocab, _ = build_vocab(["ab"] * 20)
    logits = torch.zeros(1, 3, len(vocab))
    for frame, index in enumerate([vocab["a"], vocab["<s>"], vocab["b"]]):
        logits[0, frame, index] = 5.0
    assert greedy_decode(logits, torch.tensor([3]), vocab) == ["ab"]


def test_greedy_decode_ignores_padded_frames():
    from src.models.ctc_model import greedy_decode

    vocab, _ = build_vocab(["ab"] * 20)
    a, b, blank = vocab["a"], vocab["b"], vocab["<pad>"]
    logits = torch.zeros(1, 4, len(vocab))
    for frame, index in enumerate([a, blank, b, a]):
        logits[0, frame, index] = 5.0
    assert greedy_decode(logits, torch.tensor([3]), vocab) == ["ab"]
