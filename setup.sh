#!/usr/bin/env bash
# Setup for the local PII & GIRP classifier (macOS / Linux).
# No venv/conda - installs into the Python you invoke it with.
# Usage:  bash setup.sh        (or:  PYTHON=python3.13 bash setup.sh)
set -e

PY="${PYTHON:-python3}"
echo "Using: $("$PY" --version)"

"$PY" -m pip install -r requirements.txt

echo
echo "Done. See README.md for the DataFrame and CSV examples."
echo "The model loads from local files - no internet required at run time."
