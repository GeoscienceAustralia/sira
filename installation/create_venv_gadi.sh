#!/usr/bin/env bash
set -euo pipefail

# Simple, reproducible venv creation for NCI Gadi (HOME-based)
# Usage:
#   bash installation/create_venv_gadi.sh $HOME/venv/sira

VENVDIR=${1:-"$HOME/venv/sira-env"}

module load python3/3.11.7

python3 -m venv "$VENVDIR"
source "$VENVDIR/bin/activate"
python -m pip install --upgrade pip

# Use a small cache in TMPDIR during bulk installs (optional)
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-"$TMPDIR/pip-cache"}
mkdir -p "$PIP_CACHE_DIR" || true

# Install constraints first, then core
python -m pip install -r installation/constraints.txt
python -m pip install -r installation/requirements-core.txt
python -m pip install -r installation/requirements-viz.txt
python -m pip install -r installation/requirements-dev.txt
python -m pip install -r installation/requirements-docs.txt
python -m pip install -r installation/requirements-diagrams.txt

# Build mpi4py from source only, as recommended on HPC
python3 -m pip install -v --no-binary :all: --user --cache-dir=$TMPDIR mpi4py

echo "Venv created at: $VENVDIR"
echo "To install extras later:"
echo "  source $VENVDIR/bin/activate && pip install -r installation/requirements-geo.txt"
