"""Hindi-English code-switched ASR: data pipeline and CTC training."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Keep model downloads with the project. MUCS_HF_HOME overrides this, for
# example to point at Google Drive on Colab. This must run before
# huggingface_hub is imported, which is why it lives here.
os.environ["HF_HOME"] = os.environ.get("MUCS_HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Windows consoles default to a code page that cannot print Devanagari.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")
