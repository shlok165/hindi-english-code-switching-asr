"""The Streamlit app starts, every section renders without exceptions, and the numbers are the committed ones."""

import json

import pytest

import src  # noqa: F401
from src import ROOT

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def run_section(name: str) -> AppTest:
    test = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    test.run()
    test.sidebar.radio[0].set_value(name).run()
    assert not test.exception, [e.value for e in test.exception]
    return test


@pytest.mark.parametrize("name", ["Overview", "Transcribe", "Results"])
def test_sections_render(name):
    run_section(name)


def test_overview_shows_the_final_model_numbers():
    test = run_section("Overview")
    blind = json.loads((ROOT / "results/iteration4/evaluations/blind_beam_metrics.json").read_text(encoding="utf-8"))
    values = [m.value for m in test.metric]
    assert f"{blind['wer']:.2f} %" in values and f"{blind['twer']:.2f} %" in values


def test_results_table_matches_iterations_json():
    test = run_section("Results")
    table = json.loads((ROOT / "results/iterations.json").read_text(encoding="utf-8"))
    frame = test.dataframe[0].value
    assert list(frame.index) == list(table)
    for iteration, cells in table.items():
        assert frame.loc[iteration, "blind WER"] == cells["blind_beam"]["wer"]
        assert frame.loc[iteration, "val WER (script-folded)"] == cells["val_beam"]["wer_folded"]


def test_transcribe_waits_for_an_upload():
    test = run_section("Transcribe")
    assert any("upload your own" in c.value for c in test.caption)
