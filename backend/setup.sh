#!/usr/bin/env bash
# One-shot setup: venv, deps, landmark model, tests.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
[ -d .venv ] || $PY -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
[ -f models/face_landmarker.task ] || ./scripts/get_models.sh
[ -f .env ] || cp .env.example .env
pytest -q
echo
echo "Ready. Next: source .venv/bin/activate && uvicorn facescore.api:app --reload"
