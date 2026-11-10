"""CPU inference: checkpoint checks, audio validation, greedy and ARPA beam decoding, the CLI.

A tiny random wav2vec 2.0 model stands in for the checkpoint, so these run in
seconds without the real weights. One test uses the real iteration 4 checkpoint
when it is present and is skipped otherwise.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

import src  # noqa: F401  (sets the cache location before transformers loads)
from src import ROOT
from src.arpa_lm import ArpaModel, build_arpa_decoder
from src.beam_search import BeamSettings, beam_decode, build_beam_decoder, kenlm_available, resolve_lm_backend
from src.inference import (
    ArtifactError,
    AudioLimits,
    AudioValidationError,
    InferenceError,
    Transcriber,
    check_checkpoint,
    inference_settings,
    main,
    resolve_device,
    settings_from_config,
)
from src.language_model import smoothed_model, write_arpa
from src.models.tokenizer import build_vocab

CORPUS = ["save the file", "open the file", "save the code", "नमस्ते file", "नमस्ते दोस्तों"] * 4
REAL_CHECKPOINT = ROOT / "data" / "checkpoints" / "iteration4" / "best"


# Fixtures -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """A randomly initialised two-layer wav2vec 2.0 CTC model over a small vocabulary."""
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    directory = tmp_path_factory.mktemp("checkpoint")
    vocab, _ = build_vocab(["save the file code open नमस्ते दोस्तों"] * 20)
    config = Wav2Vec2Config(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, conv_dim=(8, 8), conv_kernel=(10, 3), conv_stride=(5, 2),
        num_feat_extract_layers=2, feat_extract_norm="group", num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2, pad_token_id=vocab["<pad>"], bos_token_id=vocab["<s>"],
        eos_token_id=vocab["</s>"], apply_spec_augment=False, layerdrop=0.0,
    )
    torch.manual_seed(0)
    Wav2Vec2ForCTC(config).save_pretrained(directory)
    Wav2Vec2FeatureExtractor(sampling_rate=16000, do_normalize=True).save_pretrained(directory)
    (directory / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    return directory


@pytest.fixture(scope="module")
def tiny_lm(tmp_path_factory):
    """A 3-gram ARPA model with its unigram list, as src.language_model writes them."""
    directory = tmp_path_factory.mktemp("lm")
    arpa = directory / "lm.arpa"
    write_arpa(CORPUS, 3, arpa)
    (directory / "unigrams.txt").write_text("\n".join(sorted({w for s in CORPUS for w in s.split()})) + "\n",
                                            encoding="utf-8")
    return arpa


@pytest.fixture(scope="module")
def tiny_config(tmp_path_factory, tiny_checkpoint, tiny_lm):
    """An inference config and a decode config that point at the tiny artifacts."""
    directory = tmp_path_factory.mktemp("configs")
    decode = directory / "decode.yaml"
    decode.write_text(yaml.safe_dump({"decoder": {"kenlm_model": str(tiny_lm), "alpha": 0.5, "beta": 0.0,
                                                  "beam_width": 8}}), encoding="utf-8")
    config = directory / "inference.yaml"
    config.write_text(yaml.safe_dump({"inference": {
        "checkpoint": str(tiny_checkpoint), "decode_config": str(decode), "decoder": "greedy",
        "lm_backend": "auto", "device": "cpu", "min_seconds": 0.1, "max_seconds": 5.0, "silence_peak": 1e-4,
    }}), encoding="utf-8")
    return config


@pytest.fixture(scope="module")
def transcriber(tiny_config):
    return Transcriber.from_config(tiny_config)


def write_wav(path: Path, seconds: float = 0.5, rate: int = 16000, channels: int = 1,
              subtype: str | None = None, samples: np.ndarray | None = None) -> Path:
    if samples is None:
        generator = np.random.default_rng(0)
        samples = (0.3 * generator.standard_normal((int(seconds * rate), channels))).astype(np.float32)
    sf.write(path, samples, rate, subtype=subtype)
    return path


# Checkpoint and configuration ---------------------------------------------------------

def test_missing_checkpoint_directory_is_reported(tmp_path):
    with pytest.raises(ArtifactError, match="checkpoint directory not found"):
        check_checkpoint(tmp_path / "nope")


def test_incomplete_checkpoint_lists_the_missing_files(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactError, match="preprocessor_config.json, vocab.json, model.safetensors or"):
        check_checkpoint(tmp_path)


def test_vocab_without_special_tokens_is_rejected(tmp_path):
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "vocab.json").write_text(json.dumps({"<pad>": 0, "a": 1}), encoding="utf-8")
    with pytest.raises(ArtifactError, match="special tokens"):
        check_checkpoint(tmp_path)
    (tmp_path / "vocab.json").write_text("not json", encoding="utf-8")
    with pytest.raises(ArtifactError, match="not valid JSON"):
        check_checkpoint(tmp_path)


def test_tiny_checkpoint_passes_the_checks(tiny_checkpoint):
    vocab = check_checkpoint(tiny_checkpoint)
    assert vocab["<pad>"] == 0 and "|" in vocab


def test_vocab_size_must_match_the_model(tiny_checkpoint, tmp_path):
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        (tmp_path / name).write_bytes((tiny_checkpoint / name).read_bytes())
    vocab = json.loads((tiny_checkpoint / "vocab.json").read_text(encoding="utf-8"))
    vocab["extra"] = len(vocab)
    (tmp_path / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ArtifactError, match="outputs but vocab.json"):
        Transcriber(tmp_path, device="cpu")


def test_default_inference_config_names_the_final_model():
    settings = inference_settings()
    assert settings["checkpoint"] == "data/checkpoints/iteration4/best"
    assert settings["decode_config"] == "configs/decode_iteration4.yaml"
    assert settings["decoder"] == "greedy" and settings["device"] == "auto"


def test_missing_or_malformed_configs_are_artifact_errors(tmp_path):
    with pytest.raises(ArtifactError, match="inference config not found"):
        inference_settings(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("inference:\n  checkpoint: x\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="must set"):
        inference_settings(bad)
    with pytest.raises(ArtifactError, match="decode config not found"):
        settings_from_config(tmp_path / "missing.yaml")
    bad.write_text("decoder:\n  alpha: 0.5\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="kenlm_model"):
        settings_from_config(bad)


def test_iteration4_decode_config_loads():
    settings, kenlm = settings_from_config("configs/decode_iteration4.yaml")
    assert settings == BeamSettings(alpha=0.5, beta=0.0, unk_score_offset=-10.0, beam_width=200,
                                    beam_prune_logp=-10.0, token_min_logp=-5.0)
    assert kenlm == "data/language_model/train_5gram.klm"


def test_device_resolution():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda")
    with pytest.raises(InferenceError, match="choose auto, cpu or cuda"):
        resolve_device("tpu")
    if not torch.cuda.is_available():
        with pytest.raises(InferenceError, match="CUDA is not available"):
            resolve_device("cuda")


def test_unknown_decoder_is_rejected(tiny_checkpoint):
    with pytest.raises(InferenceError, match="choose greedy or beam"):
        Transcriber(tiny_checkpoint, decoder="viterbi", device="cpu")
    with pytest.raises(InferenceError, match="needs a decode config"):
        Transcriber(tiny_checkpoint, decoder="beam", device="cpu")


# Greedy transcription and audio validation -----------------------------------------------

def test_greedy_transcription_runs_on_cpu(transcriber, tmp_path):
    result = transcriber.transcribe(write_wav(tmp_path / "a.wav", seconds=0.5))
    assert result.decoder == "greedy" and result.lm_backend is None
    assert isinstance(result.text, str) and "<" not in result.text
    assert result.duration == pytest.approx(0.5) and result.sample_rate == 16000
    assert result.warnings == [] and result.processing_seconds >= 0
    assert set(result.to_dict()) >= {"audio", "text", "decoder", "duration", "warnings"}


def test_range_selects_part_of_the_file(transcriber, tmp_path):
    path = write_wav(tmp_path / "a.wav", seconds=2.0)
    result = transcriber.transcribe(path, start=0.5, end=1.5)
    assert (result.start, result.end, result.duration) == (0.5, 1.5, 1.0)


def test_missing_empty_and_corrupt_files(transcriber, tmp_path):
    with pytest.raises(AudioValidationError, match="not found"):
        transcriber.transcribe(tmp_path / "missing.wav")
    (tmp_path / "empty.wav").write_bytes(b"")
    with pytest.raises(AudioValidationError, match="empty \\(0 bytes\\)"):
        transcriber.transcribe(tmp_path / "empty.wav")
    (tmp_path / "corrupt.wav").write_bytes(b"RIFF\x10\x00\x00\x00WAVEfmt garbage")
    with pytest.raises(AudioValidationError, match="corrupted or unsupported"):
        transcriber.transcribe(tmp_path / "corrupt.wav")
    (tmp_path / "text.wav").write_text("this is not audio at all", encoding="utf-8")
    with pytest.raises(AudioValidationError, match="corrupted or unsupported"):
        transcriber.transcribe(tmp_path / "text.wav")


def test_too_short_and_too_long_audio(transcriber, tmp_path):
    with pytest.raises(AudioValidationError, match="too short"):
        transcriber.transcribe(write_wav(tmp_path / "short.wav", samples=np.zeros((100, 1), np.float32)))
    with pytest.raises(AudioValidationError, match="too long"):
        transcriber.transcribe(write_wav(tmp_path / "long.wav", seconds=6.0))  # the config caps at 5 s
    assert transcriber.transcribe(write_wav(tmp_path / "long.wav", seconds=6.0), end=1.0).duration == 1.0


def test_invalid_ranges(transcriber, tmp_path):
    path = write_wav(tmp_path / "a.wav", seconds=1.0)
    with pytest.raises(AudioValidationError, match="0 <= start < end"):
        transcriber.transcribe(path, start=0.8, end=0.2)
    with pytest.raises(AudioValidationError, match="past the end"):
        transcriber.transcribe(path, start=2.0)
    with pytest.raises(AudioValidationError, match="past the end"):
        transcriber.transcribe(path, end=3.0)
    assert transcriber.transcribe(path, end=1.005).end == 1.0  # manifest end times are rounded


def test_nan_and_infinite_samples_are_rejected(transcriber, tmp_path):
    bad = np.full((8000, 1), np.nan, dtype=np.float32)
    with pytest.raises(AudioValidationError, match="8000 NaN or infinite"):
        transcriber.transcribe(write_wav(tmp_path / "nan.wav", samples=bad, subtype="FLOAT"))
    bad[:4000] = np.inf
    bad[4000:] = 0.1
    with pytest.raises(AudioValidationError, match="4000 NaN or infinite"):
        transcriber.transcribe(write_wav(tmp_path / "inf.wav", samples=bad, subtype="FLOAT"))


def test_silence_gives_empty_text_and_a_warning(transcriber, tmp_path):
    result = transcriber.transcribe(write_wav(tmp_path / "silent.wav", samples=np.zeros((8000, 1), np.float32)))
    assert result.text == "" and result.warnings and "silent" in result.warnings[0]


def test_stereo_and_other_sample_rates_are_converted_with_warnings(transcriber, tmp_path):
    result = transcriber.transcribe(write_wav(tmp_path / "stereo.wav", seconds=0.5, rate=44100, channels=2))
    assert result.sample_rate == 44100
    assert any("averaged to mono" in w for w in result.warnings)
    assert any("resampled from 44100" in w for w in result.warnings)


def test_limits_can_be_changed(tiny_checkpoint, tmp_path):
    strict = Transcriber(tiny_checkpoint, device="cpu", limits=AudioLimits(min_seconds=1.0))
    with pytest.raises(AudioValidationError, match="minimum 1.0 s"):
        strict.transcribe(write_wav(tmp_path / "a.wav", seconds=0.5))


# Language model and beam search without kenlm --------------------------------------------

def test_arpa_model_matches_the_smoothed_probabilities(tiny_lm):
    probabilities, backoffs, _ = smoothed_model(CORPUS, 3)
    model = ArpaModel(tiny_lm)
    assert model.order == 3
    score, state = model.score_word(model.begin_sentence(), "save")
    assert score == pytest.approx(math.log10(probabilities[1][("<s>", "save")]), abs=1e-5)
    assert state == ("<s>", "save")
    # An unseen trigram backs off: log10 backoff(history) + log10 p(word | shorter history).
    assert ("open", "the", "code") not in probabilities[2]
    score, _ = model.score_word(("open", "the"), "code")
    expected = math.log10(backoffs.get(("open", "the"), 1.0)) + math.log10(probabilities[1][("the", "code")])
    assert score == pytest.approx(expected, abs=1e-5)
    assert "save" in model and "<unk>" not in model and "zzz" not in model
    assert model.score_word(("save",), "zzz")[1] == ()  # out of vocabulary empties the context


def test_arpa_scores_agree_with_kenlm_when_it_is_installed(tiny_lm):
    kenlm = pytest.importorskip("kenlm")
    theirs, ours = kenlm.Model(str(tiny_lm)), ArpaModel(tiny_lm)
    for sentence in ["save the file", "file the save", "नमस्ते दोस्तों file", "unknown words here", ""]:
        assert ours.score_sentence(sentence) == pytest.approx(theirs.score(sentence), abs=1e-4)
        kenlm_words = [s for s, _, _ in theirs.full_scores(sentence)]
        assert ours.word_scores(sentence) == pytest.approx(kenlm_words, abs=1e-4)


def test_lm_backend_resolution():
    assert resolve_lm_backend("arpa") == "arpa"
    assert resolve_lm_backend("auto") == ("kenlm" if kenlm_available() else "arpa")
    with pytest.raises(ValueError, match="unknown LM backend"):
        resolve_lm_backend("gpt")
    if not kenlm_available():
        with pytest.raises(RuntimeError, match="kenlm package is not installed"):
            resolve_lm_backend("kenlm")


def frames_for(text, vocab, confidence=-0.01, other=-8.0):
    ids = [vocab["|" if c == " " else c] for c in text]
    log_probs = np.full((len(ids) * 2, len(vocab)), other, dtype=np.float32)
    for t, index in enumerate(ids):
        log_probs[2 * t, index] = confidence
        log_probs[2 * t + 1, vocab["<pad>"]] = confidence
    return log_probs


def test_arpa_beam_search_recovers_text_and_the_lm_breaks_a_tie(tmp_path):
    pytest.importorskip("pyctcdecode")
    arpa = tmp_path / "lm.arpa"
    write_arpa(["save the file"] * 30, 3, arpa)
    (tmp_path / "unigrams.txt").write_text("save\nthe\nfile\n", encoding="utf-8")
    vocab, _ = build_vocab(["save the file bile"] * 20)
    settings = BeamSettings(alpha=1.0, beta=0.0)
    decoder = build_beam_decoder(vocab, arpa, settings, backend="arpa")
    assert beam_decode(frames_for("save the file", vocab), vocab, decoder, settings) == "save the file"
    tied = frames_for("save the file", vocab)
    position = 2 * "save the file".index("f")
    tied[position, vocab["f"]] = tied[position, vocab["b"]] = math.log(0.5)
    assert beam_decode(tied, vocab, decoder, settings) == "save the file"
    tied[position, vocab["f"]] = -8.0  # the acoustics now clearly say "bile", an OOV word
    assert beam_decode(tied, vocab, decoder, settings) == "save the bile"


def test_arpa_backend_and_kenlm_backend_decode_identically(tmp_path):
    pytest.importorskip("pyctcdecode")
    pytest.importorskip("kenlm")
    arpa = tmp_path / "lm.arpa"
    write_arpa(CORPUS, 3, arpa)
    (tmp_path / "unigrams.txt").write_text("\n".join({w for s in CORPUS for w in s.split()}), encoding="utf-8")
    vocab, _ = build_vocab(["save the file code open नमस्ते दोस्तों"] * 20)
    settings = BeamSettings(alpha=0.7, beta=0.5, beam_width=16)
    generator = np.random.default_rng(1)
    for _ in range(5):
        log_probs = np.log(generator.dirichlet(np.ones(len(vocab)) * 0.3, size=40)).astype(np.float32)
        decoders = [build_beam_decoder(vocab, arpa, settings, backend=b) for b in ("kenlm", "arpa")]
        outputs = [beam_decode(log_probs, vocab, d, settings) for d in decoders]
        assert outputs[0] == outputs[1]


def test_arpa_decoder_rejects_an_unknown_label_set(tiny_lm):
    pytest.importorskip("pyctcdecode")
    with pytest.raises(FileNotFoundError, match="ARPA language model not found"):
        build_beam_decoder({"<pad>": 0, "|": 1, "a": 2}, tiny_lm.with_name("absent.klm"), BeamSettings(), "arpa")
    build_arpa_decoder(["", " ", "a"], tiny_lm, ["a"], 0.5, 0.0, -10.0)


def test_transcriber_beam_decodes_with_the_arpa_backend(tiny_config, tmp_path):
    pytest.importorskip("pyctcdecode")
    transcriber = Transcriber.from_config(tiny_config, decoder="beam", lm_backend="arpa")
    assert transcriber.lm_backend == "arpa" and transcriber.settings.beam_width == 8
    result = transcriber.transcribe(write_wav(tmp_path / "a.wav", seconds=0.5))
    assert result.decoder == "beam" and result.lm_backend == "arpa"
    assert all(word in CORPUS_WORDS for word in result.text.split())


CORPUS_WORDS = {w for s in CORPUS for w in s.split()}


def test_beam_with_a_missing_language_model_is_an_artifact_error(tiny_config, tmp_path):
    pytest.importorskip("pyctcdecode")
    with pytest.raises(ArtifactError, match="not found"):
        Transcriber.from_config(tiny_config, decoder="beam", kenlm=tmp_path / "absent.klm", lm_backend="arpa")


# Command line ---------------------------------------------------------------------------

def test_cli_transcribes_and_prints_json(tiny_config, tmp_path, capsys):
    path = write_wav(tmp_path / "a.wav", seconds=0.5)
    assert main(["--audio", str(path), "--config", str(tiny_config), "--json"]) == 0
    row = json.loads(capsys.readouterr().out.strip())
    assert row["audio"] == str(path) and row["decoder"] == "greedy" and "text" in row
    assert main(["--audio", str(path), "--config", str(tiny_config)]) == 0
    assert "\n" in capsys.readouterr().out


def test_cli_reports_bad_audio_per_file_and_continues(tiny_config, tmp_path, capsys):
    good = write_wav(tmp_path / "good.wav", seconds=0.5)
    missing = tmp_path / "missing.wav"
    assert main(["--audio", str(missing), str(good), "--config", str(tiny_config), "--json"]) == 1
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert "error" in lines[0] and "not found" in lines[0]["error"]
    assert "text" in lines[1]


def test_cli_exit_codes_for_missing_artifacts_and_bad_options(tiny_config, tmp_path, capsys):
    path = write_wav(tmp_path / "a.wav", seconds=0.5)
    assert main(["--audio", str(path), "--config", str(tiny_config), "--checkpoint", str(tmp_path / "no")]) == 2
    assert "checkpoint directory not found" in capsys.readouterr().err
    assert main(["--audio", str(path), "--config", str(tmp_path / "no.yaml")]) == 2
    assert "inference config not found" in capsys.readouterr().err
    if not torch.cuda.is_available():
        assert main(["--audio", str(path), "--config", str(tiny_config), "--device", "cuda"]) == 2
        assert "CUDA is not available" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--audio", str(path), str(path), "--config", str(tiny_config), "--end", "0.2"])
    with pytest.raises(SystemExit):
        main(["--manifest", str(path), "--config", str(tiny_config)])  # --manifest needs --output
    with pytest.raises(SystemExit):
        main(["--audio", str(path), "--config", str(tiny_config), "--kenlm", "x.klm", "--decoder", "greedy"])


def test_cli_manifest_writes_a_row_per_entry(tiny_config, tmp_path):
    good = write_wav(tmp_path / "good.wav", seconds=1.0)
    manifest = tmp_path / "list.csv"
    manifest.write_text("id,audio,start,end,reference\nu1,good.wav,0,0.5,save the file\nu2,absent.wav,,,x\n"
                        "u3,good.wav,,,\n", encoding="utf-8")
    output = tmp_path / "out" / "predictions.jsonl"
    assert main(["--manifest", str(manifest), "--output", str(output), "--config", str(tiny_config)]) == 1
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["u1", "u2", "u3"]
    assert rows[0]["reference"] == "save the file" and rows[0]["duration"] == 0.5 and "hypothesis" in rows[0]
    assert rows[1]["hypothesis"] is None and "not found" in rows[1]["error"]
    assert rows[2]["duration"] == 1.0 and rows[2]["reference"] == ""


def test_cli_manifest_errors(tiny_config, tmp_path, capsys):
    assert main(["--manifest", str(tmp_path / "no.csv"), "--output", str(tmp_path / "o.jsonl"),
                 "--config", str(tiny_config)]) == 2
    assert "manifest not found" in capsys.readouterr().err
    bad = tmp_path / "bad.csv"
    bad.write_text("id,path\nu1,a.wav\n", encoding="utf-8")
    assert main(["--manifest", str(bad), "--output", str(tmp_path / "o.jsonl"), "--config", str(tiny_config)]) == 2
    assert "'audio' column" in capsys.readouterr().err


# The real model, when its checkpoint is unpacked ------------------------------------------

@pytest.mark.skipif(not REAL_CHECKPOINT.is_dir(), reason="data/checkpoints/iteration4/best is not unpacked")
def test_final_checkpoint_loads_and_transcribes_on_cpu(tmp_path):
    transcriber = Transcriber.from_config(device="cpu")
    assert len(transcriber.vocab) == transcriber.model.config.vocab_size
    rate = 16000
    t = np.arange(rate) / rate
    tone = (0.5 * np.sin(2 * np.pi * 220 * t) * np.sin(2 * np.pi * 3 * t)).astype(np.float32)
    result = transcriber.transcribe(write_wav(tmp_path / "tone.wav", samples=tone[:, None]))
    assert result.decoder == "greedy" and isinstance(result.text, str) and result.duration == 1.0
