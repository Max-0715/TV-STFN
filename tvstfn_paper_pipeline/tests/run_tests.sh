#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}

$PYTHON_BIN tvstfn_paper_pipeline/tests/smoke_test_fast.py

echo "All fast smoke tests passed."