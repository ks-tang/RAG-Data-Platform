#!/usr/bin/env bash
# Lance tout le pipeline dans l'ordre. S'arrête dès qu'une étape échoue.
set -euo pipefail
python src/ingest.py
python src/transform.py
python src/enrich.py
python src/build_chunks.py
python src/check_quality.py
