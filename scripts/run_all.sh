#!/usr/bin/env bash
# run_all.sh: End-to-end pipeline reproduction for Multimodal Meme Understanding
set -euo pipefail

echo "================================================================="
echo "  Multimodal Meme Understanding: End-to-End Pipeline & Reproduction"
echo "================================================================="

# 1. Environment and dependencies
PYTHON_BIN="${PYTHON:-python3}"
if [ -d ".venv" ]; then
    PYTHON_BIN=".venv/bin/python"
fi
echo "[1/5] Using Python environment: $($PYTHON_BIN --version) ($PYTHON_BIN)"

# 2. Data verification and holdout generation
echo "[2/5] Verifying dataset and generating group-aware hold-out split..."
$PYTHON_BIN -m src.data --images || echo "Note: If full images are not downloaded yet, run: bash scripts/download_data.sh"

# 3. Unit and contract test suite
echo "[3/5] Running contract and regression test suite..."
$PYTHON_BIN -m pytest tests/

# 4. Statistical Analysis & Paper Metrics Reproduction
echo "[4/4] Reproducing experimental analysis and statistical tests from saved results..."
$PYTHON_BIN -m src.analysis --report-date 2026-10-07
$PYTHON_BIN scripts/make_confusion_matrix.py

echo "================================================================="
echo "  Reproduction completed successfully!"
echo "  Results: results/*.json | Analysis: report/figures/"
echo "  Interactive Demo: available in notebooks/meme_understanding.ipynb (Section 7)"
echo "================================================================="
