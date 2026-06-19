#!/usr/bin/env bash
# Setup for the local PII & GIRP classifier (macOS / Linux).
# No venv/conda - installs into the Python you invoke it with.
# Usage:  bash setup.sh        (or:  PYTHON=python3.12 bash setup.sh)
set -e

PY="${PYTHON:-python3}"
echo "Using: $("$PY" --version)"

if compgen -G "wheelhouse/*.whl" > /dev/null; then
  echo "Installing from local wheelhouse only..."
  "$PY" -m pip install --no-index --find-links wheelhouse -r requirements.txt -r requirements-hybrid.txt
else
  echo "Installing from package indexes because wheelhouse was not found..."
  "$PY" -m pip install -r requirements.txt -r requirements-hybrid.txt
fi

echo
echo "Done. See README.md for the DataFrame and CSV examples."
echo "No spaCy language model download is required. The model loads from local files."
