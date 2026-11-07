#!/usr/bin/env bash
# Build the kenlm Python extension and build_binary for Windows + Python 3.14 with MinGW-w64.
# PyPI has no Windows wheel, and the bundled kenlm.cpp (Cython 0.29) does not
# compile on 3.14, so the binding is regenerated with Cython 3 first.
# Usage: scripts/build_kenlm.sh   (from the repository root, Git Bash)
set -euo pipefail
ROOT=$(pwd); PY="$ROOT/.venv/Scripts/python"
PYBASE=$("$PY" -c "import sys; print(sys.base_prefix.replace(chr(92), '/'))")
WORK=$(mktemp -d)
"$PY" -m pip install -q cython
"$PY" -m pip download -q --no-deps --no-binary :all: kenlm==0.3.0 -d "$WORK" 2>/dev/null \
  || curl -sL "$("$PY" -c "import json,urllib.request as u; d=json.load(u.urlopen('https://pypi.org/pypi/kenlm/0.3.0/json')); print([x['url'] for x in d['urls'] if x['filename'].endswith('.tar.gz')][0])")" -o "$WORK/kenlm.tar.gz"
tar xzf "$WORK"/kenlm*.tar.gz -C "$WORK"
cd "$WORK"/kenlm-0.3.0
"$PY" -m cython -3 --cplus python/kenlm.pyx -o python/kenlm.cpp
FILES=$(ls util/*.cc lm/*.cc util/double-conversion/*.cc python/*.cc | grep -vE '(main|test)\.cc$')
# -static avoids depending on MinGW runtime DLLs at import time.
g++ -O2 -DNDEBUG -DKENLM_MAX_ORDER=6 -DMS_WIN64 -std=c++17 -w -I. -I"$PYBASE/Include" -shared \
  -o kenlm.cp314-win_amd64.pyd $FILES python/kenlm.cpp -L"$PYBASE/libs" -lpython314 -static
cp kenlm.cp314-win_amd64.pyd "$ROOT/.venv/Lib/site-packages/"
# build_binary turns an ARPA file into the fast .klm format. It needs no Boost,
# unlike lmplz, which is why src/language_model.py can work without lmplz.
g++ -O2 -DNDEBUG -DKENLM_MAX_ORDER=6 -std=c++17 -w -I. -o build_binary.exe \
  lm/build_binary_main.cc $(echo "$FILES" | grep -v '^python/') -static
cp build_binary.exe "$ROOT/.venv/Scripts/"
rm -rf "$WORK"
echo "kenlm installed"
