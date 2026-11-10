"""The standalone scorer and the error-analysis exports."""

import csv
import json

import pytest

import src  # noqa: F401
from src import ROOT
from src.analysis.error_analysis import analyse, write_csv_reports
from src.metrics.official import INFERENCE_DIR, ScoringMaps
from src.score_files import (
    ScoringError,
    Utterance,
    main,
    read_pairs_csv,
    read_predictions,
    read_ref_hyp,
    resolve_twer_mode,
    score_records,
    write_outputs,
)
from src.utils.config import data_path

MAPS = ScoringMaps({"slash": "/"}, {"फाइल": "file"})
MAPS_DIR = data_path(INFERENCE_DIR)
VAL_BEAM = ROOT / "results" / "iteration4" / "evaluations" / "val_beam.jsonl"


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


# Reading -------------------------------------------------------------------------------

def test_kaldi_files_are_matched_by_id(tmp_path):
    ref = write(tmp_path / "ref.txt", "u2 open the file\nu1 save the file\n\nu3\n")
    hyp = write(tmp_path / "hyp.txt", "u1 save the file\nu3 x\nu2 open file\n")
    records = read_ref_hyp(ref, hyp)
    assert [(r.id, r.reference, r.hypothesis) for r in records] == [
        ("u2", "open the file", "open file"), ("u1", "save the file", "save the file"), ("u3", "", "x")]
    with pytest.raises(ScoringError, match="no hypothesis for 1 reference ids"):
        read_ref_hyp(ref, write(tmp_path / "short.txt", "u1 save the file\nu2 x\n"))
    with pytest.raises(ScoringError, match="ids that are not in"):
        read_ref_hyp(ref, write(tmp_path / "extra.txt", "u1 a\nu2 b\nu3 c\nu4 d\n"))
    with pytest.raises(ScoringError, match="duplicate utterance id"):
        read_ref_hyp(write(tmp_path / "dup.txt", "u1 a\nu1 b\n"), hyp)
    with pytest.raises(ScoringError, match="file not found"):
        read_ref_hyp(ref, tmp_path / "absent.txt")


def test_plain_files_are_paired_by_line(tmp_path):
    ref = write(tmp_path / "ref.txt", "save the file\nopen the file\n")
    assert [r.hypothesis for r in read_ref_hyp(ref, write(tmp_path / "hyp.txt", "a\nb\n"), with_ids=False)] == ["a", "b"]
    with pytest.raises(ScoringError, match="2 lines but"):
        read_ref_hyp(ref, write(tmp_path / "hyp3.txt", "a\nb\nc\n"), with_ids=False)


def test_csv_pairs_need_the_two_columns(tmp_path):
    pairs = write(tmp_path / "pairs.csv", "id,reference,hypothesis,duration\nu1,a b,a,1.5\n,c,c,\n")
    records = read_pairs_csv(pairs)
    assert (records[0].id, records[0].duration) == ("u1", 1.5) and (records[1].id, records[1].duration) == ("2", None)
    with pytest.raises(ScoringError, match="needs a 'hypothesis' column"):
        read_pairs_csv(write(tmp_path / "bad.csv", "reference,hyp\na,b\n"))


def test_predictions_prefer_the_raw_hypothesis_and_skip_failed_rows(tmp_path):
    rows = [
        {"id": "u1", "split": "blind", "duration": 2.0, "reference": "a", "reference_cleaned": "a",
         "hypothesis": "a /", "hypothesis_raw": "a slash"},
        {"id": "u2", "audio": "x.wav", "error": "audio file not found", "hypothesis": None, "reference": "b"},
        {"id": "u3", "audio": "y.wav", "reference": "b", "hypothesis": "b", "warnings": []},
    ]
    path = write(tmp_path / "p.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    records, info = read_predictions(path)
    assert [r.hypothesis for r in records] == ["a slash", "b"]
    assert info == {"split": "blind", "hypothesis_field": "hypothesis_raw", "failed_rows": 1}
    assert records[1].reference_cleaned is None and records[0].reference_cleaned == "a"
    with pytest.raises(ScoringError, match="no 'reference' field"):
        read_predictions(write(tmp_path / "q.jsonl", json.dumps({"id": "u", "hypothesis": "a"}) + "\n"))
    with pytest.raises(ScoringError, match="not JSON"):
        read_predictions(write(tmp_path / "r.jsonl", "{broken\n"))


def test_failed_inference_rows_are_counted_not_scored(tmp_path):
    rows = [{"id": "u1", "audio": "a.wav", "reference": "a", "hypothesis": None, "error": "too short"},
            {"id": "u2", "audio": "b.wav", "reference": "a", "hypothesis": "a"}]
    path = write(tmp_path / "p.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    records, info = read_predictions(path)
    assert len(records) == 1 and info["failed_rows"] == 1


# Scoring -------------------------------------------------------------------------------

def test_official_rules_and_error_counts():
    records = [Utterance("u1", "Save the file", "save slash file extra"),
               Utterance("u2", "", "noise"),
               Utterance("u3", "open the file", "")]
    summary, utterances, errors = score_records(records, MAPS, None, "none")
    assert summary["utterances"] == 3 and summary["scored_utterances"] == 2 and summary["empty_references"] == 1
    assert summary["empty_hypotheses"] == 1
    # u1: the -> / substitution, extra inserted (the punctuation map applied to the hypothesis);
    # u3: three deletions. 5 errors over 6 reference words.
    assert summary["counts"]["official"] == {"reference_words": 6, "errors": 5, "substitutions": 1,
                                             "deletions": 3, "insertions": 1}
    assert summary["wer"] == pytest.approx(500 / 6)
    assert summary["twer"] is None and "twer_reason" in summary
    assert [u["id"] for u in utterances] == ["u1", "u3"]
    assert utterances[0]["hypothesis"] == "save / file extra" and utterances[0]["reference"] == "save the file"
    assert [(e["id"], e["operation"], e["reference_word"], e["hypothesis_word"]) for e in errors] == [
        ("u1", "substitution", "the", "/"), ("u1", "insertion", None, "extra"),
        ("u3", "deletion", "open", None), ("u3", "deletion", "the", None), ("u3", "deletion", "file", None)]
    assert errors[0]["reference_script"] == "english_script" and errors[0]["hypothesis_script"] == "other"
    assert summary["confusions"][0] == {"operation": "deletion", "reference_word": "open", "hypothesis_word": None,
                                        "count": 1} or summary["confusions"][0]["count"] == 1


def test_twer_official_maps_the_hypothesis_only_and_approximate_maps_both():
    records = [Utterance("u1", "फाइल खोलें", "file खोलें")]
    official = score_records(records, MAPS, None, "official")[0]
    assert official["wer"] == 50.0 and official["twer"] == 50.0  # the reference keeps फाइल
    approximate = score_records(records, MAPS, None, "approximate")[0]
    assert approximate["twer_approximate"] == 0.0 and "twer" not in approximate
    without_maps = score_records(records, None, None, "official")[0]
    assert without_maps["twer"] is None and "not found" in without_maps["twer_reason"]
    assert "no punctuation mapping" in without_maps["normalisation"]["official"]


def test_twer_mode_follows_the_split():
    assert resolve_twer_mode("auto", "blind") == "official"
    assert resolve_twer_mode("auto", "test") == "approximate"
    assert resolve_twer_mode("auto", "val") == "none" and resolve_twer_mode("auto", None) == "none"
    assert resolve_twer_mode("official", "val") == "official"
    with pytest.raises(ScoringError, match="unknown T-WER mode"):
        resolve_twer_mode("maybe", "val")


def test_script_folded_wer_maps_both_sides():
    records = [Utterance("u1", "क्लिक करें", "click करें")]
    summary = score_records(records, MAPS, {"क्लिक": "click"}, "none")[0]
    assert summary["wer"] == 50.0 and summary["wer_folded"] == 0.0
    assert summary["counts"]["script_folded"]["errors"] == 0
    unavailable = score_records(records, MAPS, None, "none")[0]
    assert unavailable["wer_folded"] is None and "hi2en_train.json" in unavailable["wer_folded_reason"]


def test_cleaned_metrics_use_the_training_cleaner():
    # Training cleaning strips punctuation; official scoring keeps it as a word.
    records = [Utterance("u1", "hello , world", "hello world")]
    summary = score_records(records, MAPS, None, "none")[0]
    assert summary["wer"] == pytest.approx(100 / 3) and summary["wer_cleaned"] == 0.0
    assert summary["cer"] > 0 and summary["cer_cleaned"] == 0.0
    assert "clean_transcript" in summary["normalisation"]["cleaned"]


def test_cer_is_scored_on_the_officially_normalised_strings():
    summary = score_records([Utterance("u1", "ab", "ab"), Utterance("u2", "abcd", "abXd")], MAPS, None, "none")[0]
    assert summary["cer"] == pytest.approx(100 / 6)
    assert summary["counts"]["official_characters"] == {"reference_words": 6, "errors": 1, "substitutions": 1,
                                                        "deletions": 0, "insertions": 0}


# Writing and the command line --------------------------------------------------------------

def test_outputs_are_written_once(tmp_path):
    summary, utterances, errors = score_records([Utterance("u1", "a b", "a c")], MAPS, None, "none")
    written = write_outputs(tmp_path / "out", summary, utterances, errors)
    assert [p.name for p in written] == ["summary.json", "utterances.csv", "word_errors.csv", "confusions.csv"]
    saved = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert "confusions" not in saved and saved["top_confusions"][0]["reference_word"] == "b"
    with (tmp_path / "out" / "word_errors.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"id": "u1", "operation": "substitution", "reference_position": "1", "hypothesis_position": "1",
                     "reference_word": "b", "hypothesis_word": "c", "reference_script": "english_script",
                     "hypothesis_script": "english_script"}]
    with pytest.raises(ScoringError, match="refusing to overwrite"):
        write_outputs(tmp_path / "out", summary, utterances, errors)
    write_outputs(tmp_path / "out", summary, utterances, errors, force=True)


def test_cli_scores_kaldi_files(tmp_path, capsys):
    ref = write(tmp_path / "ref.txt", "u1 save the file\nu2 open the file\n")
    hyp = write(tmp_path / "hyp.txt", "u1 save the file\nu2 open file\n")
    out = tmp_path / "scores"
    assert main(["--ref", str(ref), "--hyp", str(hyp), "--output-dir", str(out), "--maps-dir", str(tmp_path)]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["wer"] == pytest.approx(100 / 6, abs=1e-3) and shown["twer"] is None and shown["wer_folded"] is None
    saved = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert saved["source"] == {"reference": str(ref), "hypothesis": str(hyp)} and saved["maps_dir"] is None
    assert main(["--ref", str(ref), "--hyp", str(hyp), "--output-dir", str(out)]) == 2
    assert "refusing to overwrite" in capsys.readouterr().err
    assert main(["--ref", str(tmp_path / "no.txt"), "--hyp", str(hyp), "--output-dir", str(tmp_path / "x")]) == 2
    assert "file not found" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--ref", str(ref), "--output-dir", str(tmp_path / "y")])


@pytest.mark.skipif(not (MAPS_DIR / "punct.mapping.txt").is_file(), reason="the blind scoring maps are not unpacked")
def test_rescoring_the_committed_val_results_reproduces_the_metrics(tmp_path, capsys):
    expected = json.loads(VAL_BEAM.with_name("val_beam_metrics.json").read_text(encoding="utf-8"))
    assert main(["--predictions", str(VAL_BEAM), "--output-dir", str(tmp_path / "val")]) == 0
    saved = json.loads((tmp_path / "val" / "summary.json").read_text(encoding="utf-8"))
    for key in ("wer", "cer", "wer_cleaned", "cer_cleaned", "reference_words", "scored_utterances"):
        assert saved[key] == pytest.approx(expected[key], abs=1e-9), key
    assert saved["split"] == "val" and saved["twer_mode"] == "none" and saved["twer"] is None


# Error analysis exports ---------------------------------------------------------------------

def test_error_analysis_counts_errors_by_script_and_writes_csv(tmp_path):
    rows = [{"id": "u1", "duration": 3.0, "reference": "क्लिक करें file", "hypothesis": "click करे extra"},
            {"id": "u2", "duration": 12.0, "reference": "save the file", "hypothesis": "the file"}]
    path = write(tmp_path / "p.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    report = analyse(path, {})
    # u1 aligns as three substitutions (file -> extra included); u2 deletes "save".
    assert report["by_script"]["substitutions"] == {"hindi_script -> english_script": 1,
                                                    "hindi_script -> hindi_script": 1,
                                                    "english_script -> english_script": 1}
    assert report["by_script"]["deletions"] == {"english_script": 1}
    assert report["by_script"]["insertions"] == {}
    assert report["all_confusions"] == [["क्लिक", "click", 1, "other_substitution"],
                                        ["करें", "करे", 1, "other_substitution"],  # anusvara is audible
                                        ["file", "extra", 1, "other_substitution"]]
    written = write_csv_reports(report, tmp_path / "csv")
    assert [p.name for p in written] == ["breakdown.csv", "confusions.csv", "deletions.csv", "insertions.csv", "examples.csv"]
    with (tmp_path / "csv" / "confusions.csv").open(encoding="utf-8", newline="") as handle:
        confusions = list(csv.DictReader(handle))
    assert confusions[1] == {"reference_word": "करें", "hypothesis_word": "करे", "count": "1",
                             "substitution_type": "other_substitution"}
    with (tmp_path / "csv" / "breakdown.csv").open(encoding="utf-8", newline="") as handle:
        sections = {row["section"] for row in csv.DictReader(handle)}
    assert sections >= {"error_type", "by_language", "by_switch", "by_duration", "by_script_substitutions"}
