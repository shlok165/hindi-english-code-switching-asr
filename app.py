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
    st.title("Experiment results")
    st.markdown(
        "Five training iterations, each changing one thing, compared on the **validation** split only; test and "
        "blind were decoded once per iteration. All values are percentages read from `results/iterations.json` "
        "(KenLM beam search unless the column says greedy)."
    )
    table = load_iteration_table()
    st.dataframe(table.style.format("{:.2f}").highlight_min(color="#d9efd9"), width="stretch")
    st.caption("Lowest value per column highlighted. Script-folded val WER maps English words written in Devanagari "
               "to Latin on both sides and was the measure used to choose; test T-WER is approximate because test "
               "references were not annotated for it; blind figures follow the official scoring.")
    st.subheader("WER by iteration")
    st.bar_chart(table[["val WER", "test WER", "blind WER"]], height=320)
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
