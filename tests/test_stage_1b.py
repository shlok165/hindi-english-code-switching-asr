"""Unit tests for decoding, the language model, metrics and error analysis."""

import json
import math
import random

import numpy as np
import pytest

import src  # noqa: F401
from src.analysis.error_analysis import (
    analyse,
    consonant_skeleton,
    homophone_key,
    substitution_type,
    switch_context,
)
from src.beam_search import BeamSettings, beam_decode, build_beam_decoder, greedy_from_log_probs
from src.data.dataset import LengthBatchSampler
from src.language_model import smoothed_model, write_arpa
from src.metrics.cer import count_characters, score as score_cer
from src.metrics.official import ScoringMaps, apply_map, approximate_twer, official_wer, read_mapping
from src.metrics.wer import count
from src.models.tokenizer import build_vocab

MAPS = ScoringMaps({"slash": "/"}, {"फाइल": "file"})
CORPUS = ["save the file", "open the file", "save the code", "नमस्ते file", "नमस्ते दोस्तों"] * 4


def frames_for(text, vocab, confidence=-0.01, other=-8.0):
    ids = [vocab["|" if c == " " else c] for c in text]
    log_probs = np.full((len(ids) * 2, len(vocab)), other, dtype=np.float32)
    for t, index in enumerate(ids):
        log_probs[2 * t, index] = confidence
        log_probs[2 * t + 1, vocab["<pad>"]] = confidence
    return log_probs


def test_greedy_keeps_real_double_letters_and_drops_blanks():
    vocab, _ = build_vocab(["aab"] * 20)
    assert greedy_from_log_probs(frames_for("aab", vocab), vocab) == "aab"


def test_greedy_matches_a_repeated_frame_run():
    vocab, _ = build_vocab(["ab"] * 20)
    a, b = vocab["a"], vocab["b"]
    log_probs = np.full((4, len(vocab)), -8.0)
    for frame, index in enumerate([a, a, b, b]):
        log_probs[frame, index] = 0.0
    assert greedy_from_log_probs(log_probs, vocab) == "ab"


def test_arpa_probabilities_sum_to_one():
    probabilities, backoffs, _ = smoothed_model(CORPUS, 3)
    words = [g[0] for g in probabilities[0]]

    def p(word, history):
        for level in range(len(history), -1, -1):
            suffix = history[len(history) - level :]
            if level + 1 <= len(probabilities) and suffix + (word,) in probabilities[level]:
                value = probabilities[level][suffix + (word,)]
                for skip in range(len(history) - level):
                    value *= backoffs.get(history[skip:], 1.0)
                return value
        raise KeyError(word)

    assert sum(probabilities[0].values()) == pytest.approx(1.0, abs=1e-9)
    for history in [("the",), ("save", "the"), ("<s>",)]:
        assert sum(p(word, history) for word in words) == pytest.approx(1.0, abs=1e-9)


def test_kenlm_reads_the_arpa_and_agrees_with_the_model(tmp_path):
    kenlm = pytest.importorskip("kenlm")
    path = tmp_path / "lm.arpa"
    write_arpa(CORPUS, 3, path)
    probabilities, _, _ = smoothed_model(CORPUS, 3)
    model = kenlm.Model(str(path))
    assert model.order == 3
    state, next_state = kenlm.State(), kenlm.State()
    model.BeginSentenceWrite(state)
    score = model.BaseScore(state, "save", next_state)
    assert score == pytest.approx(math.log10(probabilities[1][("<s>", "save")]), abs=1e-4)


def test_beam_search_recovers_text_and_lm_breaks_an_acoustic_tie(tmp_path):
    pytest.importorskip("pyctcdecode")
    pytest.importorskip("kenlm")
    arpa = tmp_path / "lm.arpa"
    write_arpa(["save the file"] * 30, 3, arpa)
    (tmp_path / "unigrams.txt").write_text("save\nthe\nfile\n", encoding="utf-8")
    vocab, _ = build_vocab(["save the file bile"] * 20)
    settings = BeamSettings(alpha=1.0, beta=0.0)
    decoder = build_beam_decoder(vocab, arpa, settings)
    assert beam_decode(frames_for("save the file", vocab), vocab, decoder, settings) == "save the file"

    tied = frames_for("save the file", vocab)
    position = 2 * "save the file".index("f")
    tied[position, vocab["f"]] = tied[position, vocab["b"]] = math.log(0.5)
    assert beam_decode(tied, vocab, decoder, settings) == "save the file"


def test_beam_search_never_emits_hidden_tokens(tmp_path):
    pytest.importorskip("pyctcdecode")
    pytest.importorskip("kenlm")
    arpa = tmp_path / "lm.arpa"
    write_arpa(["a b"] * 30, 2, arpa)
    (tmp_path / "unigrams.txt").write_text("a\nb\n", encoding="utf-8")
    vocab, _ = build_vocab(["a b"] * 20)
    settings = BeamSettings()
    decoder = build_beam_decoder(vocab, arpa, settings)
    log_probs = frames_for("a b", vocab)
    log_probs[1, vocab["<unk>"]] = 0.0
    assert "<" not in beam_decode(log_probs, vocab, decoder, settings)


def test_official_rules_map_hypothesis_only():
    result = official_wer([("file a / b", "फाइल a slash b")], MAPS)
    assert result["wer"] == 25.0
    assert result["twer"] == 0.0


def test_official_scoring_skips_empty_references_but_not_empty_hypotheses():
    result = official_wer([("", "extra words"), ("a b", "")], MAPS)
    assert result["scored_utterances"] == 1
    assert result["wer"] == 100.0


def test_official_scoring_lower_cases_both_sides():
    assert official_wer([("Git Status", "GIT status")], MAPS)["wer"] == 0.0


def test_approximate_twer_maps_unannotated_references_too():
    assert approximate_twer([("फाइल a", "file a")], MAPS)["twer_approximate"] == 0.0
    assert official_wer([("फाइल a", "file a")], MAPS)["twer"] == 50.0


def test_first_mapping_entry_wins_like_the_official_scorer(tmp_path):
    path = tmp_path / "map.txt"
    path.write_text("क one\nक two\nख three words\n", encoding="utf-8")
    assert read_mapping(path) == {"क": "one", "ख": "three words"}
    assert apply_map("क ख", read_mapping(path)) == "one three words"


def test_cer_edit_total_agrees_with_the_slow_alignment():
    rng = random.Random(0)
    for _ in range(200):
        reference = "".join(rng.choices("abक ", k=rng.randint(0, 20)))
        hypothesis = "".join(rng.choices("abक ", k=rng.randint(0, 20)))
        assert count_characters(reference, hypothesis).errors == count(list(reference), list(hypothesis)).errors


def test_cer_without_cleaning_keeps_symbols():
    assert score_cer([("a.b", "ab")], clean=False).counts.deletions == 1
    assert score_cer([("a.b", "a b")]).counts.errors == 0


def test_sampler_can_force_one_utterance_per_batch():
    lengths = np.random.default_rng(0).integers(8000, 160000, 200)
    sampler = LengthBatchSampler(lengths, 1_600_000, seed=1, max_utterances=1)
    batches = sampler.batches(0)
    assert all(len(b) == 1 for b in batches)
    assert sorted(b[0] for b in batches) == list(range(200))


def test_skeletons_match_across_scripts():
    assert consonant_skeleton("file") == consonant_skeleton("फाइल")
    assert consonant_skeleton("computer") == consonant_skeleton("कंप्यूटर".replace("ं", "म"))


def test_homophone_key_folds_vowel_length_and_nukta():
    assert homophone_key("जरूर") == homophone_key("ज़रुर")
    assert homophone_key("गयी") == homophone_key("गई")
    assert homophone_key("किये") == homophone_key("किए")
    assert homophone_key("दिन") != homophone_key("दीन".replace("द", "ब"))


def test_substitution_types():
    words = {"फाइल": "file"}
    assert substitution_type("file", "फाइल", words) == "script_only"
    assert substitution_type("file", "फ़ाइल", {}) == "incorrect_transliteration"
    assert substitution_type("जरूर", "ज़रुर", {}) == "homophone_confusion"
    assert substitution_type("file", "घर", {}) == "other_substitution"


def test_switch_direction_is_read_from_the_reference():
    words = "यह file है".split()
    assert switch_context(words, 1, {}) == "hi_to_en"
    assert switch_context(words, 2, {}) == "en_to_hi"
    assert switch_context("यह है".split(), 0, {}) is None


def test_mapped_devanagari_english_counts_as_english():
    assert switch_context("यह फाइल है".split(), 1, {"फाइल": "file"}) == "hi_to_en"


def test_analysis_ignores_empty_references_and_counts_insertions(tmp_path):
    path = tmp_path / "p.jsonl"
    rows = [{"id": "a", "reference": "", "hypothesis": "x", "duration": 2.0},
            {"id": "b", "reference": "यह file है", "hypothesis": "यह फाइल है तो", "duration": 25.0}]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    report = analyse(path, {"फाइल": "file"})
    assert report["utterances"] == 1
    assert report["error_types"]["insertion"] == 1
    assert report["error_types"]["script_only"] == 1
    assert report["by_duration"]["over 20 s"]["errors"] == 2
    assert report["wer"] == pytest.approx(66.67)


def test_checkpoint_average_is_the_mean_and_keeps_integer_tensors():
    import torch

    from src.training.average_checkpoints import average_state_dicts

    first = {"w": torch.tensor([1.0, 2.0], dtype=torch.float16), "n": torch.tensor([3])}
    second = {"w": torch.tensor([3.0, 6.0], dtype=torch.float16), "n": torch.tensor([9])}
    averaged = average_state_dicts([first, second])
    assert averaged["w"].dtype == torch.float16
    assert averaged["w"].tolist() == [2.0, 4.0]
    assert averaged["n"].tolist() == [3]
    with pytest.raises(ValueError):
        average_state_dicts([first, {"w": first["w"]}])


def test_cmudict_spellings_reach_common_transcriptions():
    from src.data.script_normalisation import fold, spellings

    cases = {
        "क्लिक": ["K", "L", "IH1", "K"],
        "फाइल": ["F", "AY1", "L"],
        "प्रोजेक्ट": ["P", "R", "AA1", "JH", "EH0", "K", "T"],
        "टर्मिनल": ["T", "ER1", "M", "AH0", "N", "AH0", "L"],
        "वीडियो": ["V", "IH1", "D", "IY0", "OW2"],
    }
    for devanagari, phones in cases.items():
        assert fold(devanagari) in {fold(s) for s in spellings(phones)}, devanagari


def test_short_words_need_strong_context_and_plurals_must_agree():
    from src.data.script_normalisation import accepted, keeps_plural, normalise

    assert not accepted("key", 0.0, 0.11)       # की, the Hindi "of"
    assert accepted("click", 0.0, 0.97)
    assert accepted("update", 0.0, 0.11)        # long words need less context evidence
    assert not accepted("monitor", 0.14, 0.19)  # मानचित्र: close spelling, weak context
    assert not keeps_plural("सिस्टम्स", "system")
    assert keeps_plural("मैसेज", "message")
    assert normalise("इस फाइल को सेव करें", {"फाइल": "file"}) == "इस file को सेव करें"


def test_edge_errors_separate_segment_edges_from_the_middle():
    from src.analysis.boundary_errors import edge_errors

    result = edge_errors("a b c d".split(), "x a b c d e".split())
    assert result["insertions"] == 2 and result["edge_insertions"] == 2  # x at the start, e at the end
    inside = edge_errors("a b c d e".split(), "a b d e".split())
    assert inside["deletions"] == 1 and inside["edge_deletions"] == 0   # c is inside
    assert edge_errors("a b c".split(), "b c".split())["edge_deletions"] == 1


def test_misalignment_filter_ignores_script_and_spelling_differences():
    from src.data.filter_train import Skeleton, misaligned

    skeleton = Skeleton({
        "console": [["K", "AA1", "N", "S", "OW0", "L"]],
        "window": [["W", "IH1", "N", "D", "OW0"]],
        "type": [["T", "AY1", "P"]],
        "acetate": [["AE1", "S", "AH0", "T", "EY2", "T"]],
    })
    assert skeleton("कंसोल विंडो पर टाइप करें") == skeleton("console window पर type करें")
    # Same words, different script: kept.
    assert not misaligned("कंसोल विंडो पर टाइप करें", "console window पर type करें", skeleton)
    # A spelling CMUdict lacks: the skeletons differ, but as written it is close, so kept.
    assert not misaligned("molar flow mixture ethyl acetate", "molar flow mixture athyl acitate", skeleton)
    # A different sentence: dropped.
    assert misaligned("हम method को class से access कर सकते हैं", "यो के कारण पूरा होता है", skeleton)
    # Most of the transcript missing from the audio: dropped.
    assert misaligned("और यह हमारा return statement है", "अ और हैं", skeleton)


def conv_frames(samples):
    """Frames out of the wav2vec 2.0 feature encoder for this many samples."""
    for kernel, stride in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
        samples = (samples - kernel) // stride + 1
    return samples


def test_realignment_windows_cover_every_frame_once():
    from src.data.realign import HOP_SAMPLES, SAMPLES_PER_FRAME, stitch, window_starts, WINDOW_SAMPLES

    for samples in (100_000, WINDOW_SAMPLES, WINDOW_SAMPLES + 1, 9_600_123, 34_272_000):
        windows = [np.arange(conv_frames(min(samples, start + WINDOW_SAMPLES) - start)) + start // SAMPLES_PER_FRAME
                   for start in window_starts(samples)]
        assert all(start % SAMPLES_PER_FRAME == 0 for start in window_starts(samples))
        assert np.array_equal(stitch(windows), np.arange(conv_frames(samples)))
        assert samples - window_starts(samples)[-1] >= min(samples, WINDOW_SAMPLES - HOP_SAMPLES)


def test_joining_delimiter_may_sit_with_either_neighbour():
    from src.data.realign import ctc_states, state_bounds

    states, owners = ctc_states([[5, 6], [7, 8]], blank=0, delimiter=4)
    assert states.tolist() == [0, 5, 0, 6, 0, 4, 0, 7, 0, 8, 0]
    assert owners.tolist() == [0, 0, -1, 1, 1]
    lower, upper = state_bounds(owners, np.array([0, 10]), np.array([12, 30]), frames=40)
    assert (lower[1], upper[1]) == (0, 12)      # first transcript's label
    assert (lower[5], upper[5]) == (0, 30)      # the delimiter: either neighbour's range
    assert (lower[7], upper[7]) == (10, 30)     # second transcript's label
    assert (lower[-1], upper[-1]) == (10, 39)   # final blank runs to the last frame
    assert np.all(np.diff(lower) >= 0) and np.all(np.diff(upper) >= 0)


def test_realignment_moves_the_cut_into_the_pause():
    from src.data.realign import FRAME_SECONDS, realign_recording

    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "a": 5, "b": 6, "c": 7, "d": 8}
    # Frames: "ab", a long pause (frames 6-21) with the joining "|" at 12, then "cd".
    best = [0, 0, 0, 5, 0, 6] + [0] * 6 + [4] + [0] * 9 + [7, 7, 0, 8, 0, 0]
    frames = np.full((len(best), len(vocab)), -8.0)
    frames[np.arange(len(best)), best] = -0.01
    # The released cut is at frame 4, between "a" and "b", so "b" sits in the wrong segment.
    rows = [{"start": 0.0, "end": 4 * FRAME_SECONDS, "text": "ab"},
            {"start": 4 * FRAME_SECONDS, "end": len(best) * FRAME_SECONDS, "text": "cd"}]
    new = realign_recording(rows, frames, vocab, margin=1.0)
    cut = (5 + 1 + 22) / 2 * FRAME_SECONDS    # middle of the pause between "b" and "c"
    assert new == [(0.0, round(cut, 3)), (round(cut, 3), round(len(best) * FRAME_SECONDS, 3))]
    # A silent pause is split at its middle, however little padding is allowed.
    assert realign_recording(rows, frames, vocab, margin=1.0, pad=0.1) == new
    # A pause holding speech no transcript accounts for (an "a" at frame 9) belongs to
    # neither segment: each cut stays within the padding of its transcript's speech.
    spoken = frames.copy()
    spoken[9], spoken[9, 5] = -8.0, -0.01
    assert realign_recording(rows, spoken, vocab, margin=1.0, pad=0.1) == [(0.0, 0.22), (0.34, 0.56)]
    # With no margin, four letters cannot fit in a segment three frames long.
    short = [{"start": 0.0, "end": 2 * FRAME_SECONDS, "text": "abab"},
             {"start": 2 * FRAME_SECONDS, "end": len(best) * FRAME_SECONDS, "text": "cd"}]
    assert realign_recording(short, frames, vocab, margin=0.0) is None
    assert realign_recording(short, frames, vocab, margin=1.0) is not None


def test_cut_effects_counts_errors_by_position_and_classes_released_cuts():
    from src.analysis.cut_effects import PLACES, classify, error_positions
    from src.data.realign import FRAME_SECONDS

    positions = error_positions([("a b c d", "x b c"), ("e", "f"), ("", "g")])
    assert positions["reference_words"] == 5 and positions["wer"] == 60.0
    assert positions[PLACES[0]]["substitutions"] == 1   # a -> x, before the first correct word
    assert positions[PLACES[2]]["deletions"] == 1       # d, after the last correct word
    assert positions[PLACES[3]]["substitutions"] == 1   # e -> f, with no word right
    assert positions[PLACES[1]]["wer_points"] == 0.0

    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "a": 5, "b": 6, "c": 7, "d": 8}
    best = [0, 0, 0, 5, 0, 6] + [0] * 6 + [4] + [0] * 9 + [7, 7, 0, 8, 0, 0]
    frames = np.full((len(best), len(vocab)), -8.0)
    frames[np.arange(len(best)), best] = -0.01

    def clean(cut, scores=frames):
        rows = [{"id": "x", "start": 0.0, "end": cut * FRAME_SECONDS, "text": "ab"},
                {"id": "y", "start": cut * FRAME_SECONDS, "end": len(best) * FRAME_SECONDS, "text": "cd"}]
        found = classify(rows, scores, vocab, tolerance=0.01, stray_limit=1)
        return [found["x"]["clean"], found["y"]["clean"]]

    assert clean(4) == [False, False]   # cut between "a" and "b"
    assert clean(12) == [True, True]    # cut in the silent pause
    spoken = frames.copy()
    spoken[9], spoken[9, 5] = -8.0, -0.01
    assert clean(12, spoken) == [False, True]   # untranscribed "a" inside the first segment
