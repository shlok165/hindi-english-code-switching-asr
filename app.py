"""Streamlit dashboard for the Hindi-English code-switched ASR project.

Launch from the repository root:

    .venv\\Scripts\\python -m streamlit run app.py

Three sections, chosen in the sidebar:
- Overview:   what the project is and the headline result.
- Transcribe: upload a WAV file, play it, transcribe it on the CPU with the final model (iteration 4).
- Results:    the committed metrics of the five training iterations.

Everything shown comes from the existing backend (`src.inference`) and the committed result files
under `results/`. Nothing here trains a model or needs a GPU. Each section is one function, so a new
section is a new function plus one entry in SECTIONS.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import ArtifactError, AudioValidationError, InferenceError, Transcriber, Transcription

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FINAL_ITERATION = "iteration4"
DECODER_LABELS = {"greedy": "Greedy CTC (no language model)", "beam": "KenLM beam search (5-gram LM)"}


# --------------------------------------------------------------------------- data access

@st.cache_data(show_spinner=False)
def load_iteration_table() -> pd.DataFrame:
    """One row per iteration from results/iterations.json, with the columns people compare.

    The JSON holds, per iteration, one cell per split and decoder (e.g. "blind_beam") with
    wer, cer and, where defined, twer (blind), twer_approximate (test) or wer_folded (val).
    """
    table = json.loads((RESULTS / "iterations.json").read_text(encoding="utf-8"))
    rows = []
    for iteration, cells in table.items():
        rows.append({
            "iteration": iteration,
            "val WER": cells["val_beam"]["wer"],
            "val WER (script-folded)": cells["val_beam"]["wer_folded"],
            "test WER": cells["test_beam"]["wer"],
            "test T-WER (approx.)": cells["test_beam"]["twer_approximate"],
            "blind WER": cells["blind_beam"]["wer"],
            "blind T-WER": cells["blind_beam"]["twer"],
            "blind CER": cells["blind_beam"]["cer"],
            "blind WER, greedy": cells["blind_greedy"]["wer"],
        })
    return pd.DataFrame(rows).set_index("iteration")


@st.cache_data(show_spinner=False)
def load_metrics(iteration: str, split: str, decoder: str) -> dict:
    """The *_metrics.json that src.evaluate wrote for one split and decoder."""
    directory = RESULTS / "evaluations" if iteration == "iteration1" else RESULTS / iteration / "evaluations"
    return json.loads((directory / f"{split}_{decoder}_metrics.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading the model (and, for beam search, the language model)…")
def get_transcriber(decoder: str) -> Transcriber:
    """The final model, loaded once per decoder and kept for the life of the server.

    Transcriber.from_config reads configs/inference.yaml: checkpoint, decode settings and limits.
    Raises ArtifactError when the checkpoint or language model is not unpacked.
    """
    return Transcriber.from_config(decoder=decoder, device="cpu")


def transcribe_upload(upload, decoder: str) -> Transcription:
    """Write the uploaded bytes to a temporary file and run the backend's validation and model on it."""
    suffix = Path(upload.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(upload.getbuffer())
        path = Path(handle.name)
    try:
        return get_transcriber(decoder).transcribe(path)
    finally:
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- results visuals

# Dark-surface categorical hues, validated colorblind-safe against Streamlit's
# chart surface (#0e1117). "Blind WER" keeps aqua in every chart it appears in, so
# the same series is always the same colour.
RESULTS_COLORS = {
    "Validation WER": "#3987e5",  # blue
    "Test WER": "#d95926",        # orange
    "Blind WER": "#199e70",       # aqua
    "Blind T-WER": "#c98500",     # yellow
    "Blind CER": "#199e70",       # aqua (single series)
}


def render_grouped_chart(
    table: pd.DataFrame,
    series: dict[str, str],
    y_title: str,
    y_max: float,
    key: str,
    alt: str,
) -> None:
    """Render iteration metrics as grouped bars on a zero-based scale."""
    rows = [
        {"Iteration": iteration.removeprefix("iteration"), "Metric": label, "Value": table.loc[iteration, column]}
        for iteration in table.index
        for label, column in series.items()
    ]
    labels = list(series)
    chart = {
        "mark": {"type": "bar", "cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3},
        "encoding": {
            "x": {
                "field": "Iteration",
                "type": "ordinal",
                "sort": ["1", "2", "3", "4", "5"],
                "axis": {"title": "Iteration", "labelAngle": 0, "labelPadding": 8},
            },
            "xOffset": {"field": "Metric", "type": "nominal"},
            "y": {
                "field": "Value",
                "type": "quantitative",
                "scale": {"domain": [0, y_max], "zero": True, "nice": False},
                "axis": {"title": y_title, "grid": True, "format": ".0f"},
            },
            "color": {
                "field": "Metric",
                "type": "nominal",
                "scale": {"domain": labels, "range": [RESULTS_COLORS[label] for label in labels]},
                "legend": {
                    "orient": "top",
                    "direction": "horizontal",
                    "title": None,
                    "labelLimit": 220,
                    "symbolType": "square",
                    "symbolSize": 110,
                } if len(labels) > 1 else None,
            },
            "tooltip": [
                {"field": "Iteration", "type": "ordinal", "title": "Iteration"},
                {"field": "Metric", "type": "nominal", "title": "Measure"},
                {"field": "Value", "type": "quantitative", "title": y_title, "format": ".2f"},
            ],
        },
        "height": 300,
        "config": {
            "view": {"stroke": None},
            "axis": {"labelFontSize": 12, "titleFontSize": 12, "titleFontWeight": 600},
            "legend": {"labelFontSize": 12, "padding": 4},
        },
    }
    st.vega_lite_chart(
        pd.DataFrame(rows), chart, width="stretch", theme="streamlit", key=key, alt=alt
    )


# --------------------------------------------------------------------------- sections

def render_overview() -> None:
    st.title("Hindi-English code-switched speech recognition")
    st.markdown(
        "Hindi speakers switch to English mid-sentence, especially for technical terms, so a recogniser has to "
        "handle two languages and two scripts in one utterance. This project fine-tunes **wav2vec 2.0** "
        "(Vakyansh Hindi checkpoint) with a **CTC** head on **MUCS 2021 Subtask-2** (spoken-tutorial recordings, "
        "85.6 h of training speech) and decodes with greedy search or **KenLM beam search** over a 5-gram model "
        "built from the training transcripts."
    )
    blind = load_metrics(FINAL_ITERATION, "blind", "beam")
    val = load_metrics(FINAL_ITERATION, "val", "beam")
    c1, c2, c3 = st.columns(3)
    c1.metric("Blind-test WER", f"{blind['wer']:.2f} %", help="Official MUCS scoring, confirmed with the organisers' script")
    c2.metric("Blind-test T-WER", f"{blind['twer']:.2f} %", help="Transliteration-aware WER, the second official metric")
    c3.metric("Validation WER", f"{val['wer']:.2f} %", help="The tuning split, held out from training")
    st.caption(f"Final model: {FINAL_ITERATION} with KenLM beam search, chosen on validation data only. "
               "Published references on the same blind set: official baseline 25.53 WER, Chadha et al. (2022) 21.77.")
    st.markdown(
        """
**How it was built**

1. Data pipeline and a 120-character tokenizer (Devanagari, Latin, digits, spoken symbols); text keeps its script.
2. Baseline fine-tuning on an RTX 3050 (iteration 1), KenLM decoding tuned on validation, blind result confirmed
   with the organisers' `compute_wer.py`.
3. Iterations 2-5, one change each (more training, script normalisation, a misalignment filter, realignment),
   compared on validation only. Iteration 4 is the final model.
4. CPU inference and scoring tools, and this dashboard.

See `README.md`, `docs/backend.md` and `docs/iterations.md` for the details.
"""
    )


def render_transcription() -> None:
    st.title("Transcribe a recording")
    st.markdown("Upload a WAV file (any sample rate; 0.1 s to 60 s). It is transcribed on the CPU by the final model.")
    decoder = st.radio("Decoder", list(DECODER_LABELS), format_func=DECODER_LABELS.get, horizontal=True)
    upload = st.file_uploader("WAV file", type=["wav"])
    if upload is None:
        st.caption("No recording ships with the repository (MUCS audio is not committed); upload your own to begin.")
        return
    st.audio(upload)

    try:
        with st.spinner("Transcribing on the CPU…"):
            result = transcribe_upload(upload, decoder)
    except ArtifactError as error:  # checkpoint or language model not unpacked
        st.error(f"Model files are missing: {error}\n\nUnpack `checkpoints.zip` in the repository root; see docs/cpu_setup.md.")
        return
    except AudioValidationError as error:  # empty, corrupt, silent-only, too short/long, non-finite audio
        st.error(f"This file cannot be transcribed: {error}")
        return
    except InferenceError as error:  # anything else the backend reports, e.g. pyctcdecode missing for beam search
        st.error(str(error))
        return

    st.subheader("Transcript")
    if result.text:
        st.markdown(f"<p style='font-size:1.5rem; line-height:1.6'>{result.text}</p>", unsafe_allow_html=True)
    else:
        st.warning("The model produced no text for this recording.")
    for warning in result.warnings:
        st.warning(warning)
    c1, c2, c3 = st.columns(3)
    c1.metric("Audio length", f"{result.duration:.2f} s")
    c2.metric("Processing time", f"{result.processing_seconds:.2f} s")
    c3.metric("Language model", result.lm_backend or "none")
    with st.expander("JSON, as `python -m src.inference --json` prints it"):
        st.json({**result.to_dict(), "audio": upload.name})


def render_results() -> None:
    st.markdown(
        """
<style>
[data-testid="stAppViewContainer"] .block-container {
    max-width: 1480px;
    padding-top: 2rem;
    padding-bottom: 3.5rem;
}
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    font-family: "Bahnschrift", "Segoe UI", sans-serif;
    letter-spacing: 0;
}
.results-kicker {
    color: #72c9bd;
    font-size: 0.76rem;
    font-weight: 700;
    line-height: 1.4;
    text-transform: uppercase;
}
.results-intro {
    max-width: 900px;
    line-height: 1.6;
}
</style>
<div class="results-kicker">Model evaluation · Iterations 1–5</div>
""",
        unsafe_allow_html=True,
    )
    st.title("Experiment results")
    st.markdown(
        "<div class='results-intro'>Five controlled training iterations, evaluated with KenLM beam search. "
        "Validation alone determined model selection; test and blind results are reported for comparison.</div>",
        unsafe_allow_html=True,
    )
    table = load_iteration_table()

    with st.container(border=True, key="iteration4-selection"):
        st.markdown("#### Selected model · Iteration 4")
        selection_metrics = st.columns(3, gap="medium")
        selection_metrics[0].metric("Validation WER · script-folded", f"{table.loc[FINAL_ITERATION, 'val WER (script-folded)']:.2f}%")
        selection_metrics[1].metric("Validation WER · raw", f"{table.loc[FINAL_ITERATION, 'val WER']:.2f}%")
        selection_metrics[2].metric("Test WER", f"{table.loc[FINAL_ITERATION, 'test WER']:.2f}%")
        selection_metrics = st.columns(3, gap="medium")
        selection_metrics[0].metric("Blind WER", f"{table.loc[FINAL_ITERATION, 'blind WER']:.2f}%")
        selection_metrics[1].metric("Blind T-WER", f"{table.loc[FINAL_ITERATION, 'blind T-WER']:.2f}%")
        selection_metrics[2].metric("Blind CER", f"{table.loc[FINAL_ITERATION, 'blind CER']:.2f}%")
        best_folded = table["val WER (script-folded)"].min()
        st.markdown(
            f"Chosen on validation only: iteration 4 has the lowest script-folded validation WER "
            f"({best_folded:.2f}%), narrowly ahead of iteration 2 ({table.loc['iteration2', 'val WER (script-folded)']:.2f}%). "
            "Test and blind metrics did not determine the selection."
        )

    st.subheader("WER across splits")
    render_grouped_chart(
        table,
        {"Validation WER": "val WER", "Test WER": "test WER", "Blind WER": "blind WER"},
        "WER (%)",
        40,
        "iteration-wer-chart",
        "Grouped validation, test, and blind WER by iteration; vertical scale starts at zero.",
    )
    st.caption(
        "Validation WER for iteration 3 uses its script-normalized references; the script-folded validation "
        "measure below is the consistent cross-iteration comparison."
    )

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.subheader("Blind WER and T-WER")
        render_grouped_chart(
            table,
            {"Blind WER": "blind WER", "Blind T-WER": "blind T-WER"},
            "Error rate (%)",
            25,
            "blind-wer-twer-chart",
            "Grouped blind WER and T-WER by iteration; vertical scale starts at zero.",
        )
        st.caption("Blind T-WER follows the official transliteration-aware scoring.")
    with right:
        st.subheader("Blind CER")
        render_grouped_chart(
            table,
            {"Blind CER": "blind CER"},
            "CER (%)",
            16,
            "blind-cer-chart",
            "Blind CER by iteration; vertical scale starts at zero.",
        )

    st.subheader("All reported metrics")
    st.dataframe(
        table.style.format("{:.2f}").highlight_min(props="background-color:#d9efd9; color:#0b0b0b;"),
        width="stretch",
        height=300,
        row_height=42,
        column_config={
            "_index": st.column_config.TextColumn("Iteration", width="small"),
            "val WER": st.column_config.NumberColumn("Validation WER", format="%.2f", width="small"),
            "val WER (script-folded)": st.column_config.NumberColumn("Validation WER · folded", format="%.2f", width="medium"),
            "test WER": st.column_config.NumberColumn("Test WER", format="%.2f", width="small"),
            "test T-WER (approx.)": st.column_config.NumberColumn("Test T-WER · approx.", format="%.2f", width="medium"),
            "blind WER": st.column_config.NumberColumn("Blind WER", format="%.2f", width="small"),
            "blind T-WER": st.column_config.NumberColumn("Blind T-WER", format="%.2f", width="small"),
            "blind CER": st.column_config.NumberColumn("Blind CER", format="%.2f", width="small"),
            "blind WER, greedy": st.column_config.NumberColumn("Blind WER · greedy", format="%.2f", width="medium"),
        },
        hide_index=False,
        alt="All committed evaluation metrics for iterations 1 through 5, in percent.",
    )
    st.caption("All values are percentages. Lowest values per table column are highlighted.")

    with st.expander("Why raw and script-folded WER differ"):
        st.markdown(
            "Raw WER compares the written words exactly, so an English word written in Devanagari on one side "
            "and Latin on the other counts as an error. Script-folded WER first maps known English-in-Devanagari "
            "forms to Latin on both references and hypotheses using a map built from training data only. It reduces "
            "script-only mismatches; it does not change the underlying audio or correct other word errors."
        )

    with st.expander("What changed in each iteration"):
        st.markdown(
            """
| Iteration | Change |
|---|---|
| 1 | Baseline: wav2vec 2.0 him-4200 + CTC, 38,000 updates |
| 2 | More training: 44,000 further updates from iteration 1 at half the peak learning rate |
| 3 | Script normalisation: English words in Devanagari mapped to Latin in the training text and LM |
| 4 | Misalignment filter: training utterances whose transcript does not match the audio dropped (final model) |
| 5 | Realignment: those segments re-cut where their transcripts are, instead of dropped |
"""
        )


SECTIONS = {
    "Overview": render_overview,
    "Transcribe": render_transcription,
    "Results": render_results,
}


def main() -> None:
    st.set_page_config(page_title="Hindi-English ASR", page_icon="🎙", layout="wide")
    with st.sidebar:
        st.markdown("**Hindi-English ASR**")
        section = st.radio("Section", list(SECTIONS))
        st.caption("Metrics come from the committed `results/` files; inference runs on the CPU with iteration 4.")
    SECTIONS[section]()


if __name__ == "__main__":
    main()
