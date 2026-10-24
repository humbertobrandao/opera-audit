#!/usr/bin/env bash
# Full reproduction, open track (no licence needed).
# Builds the environment of record and reruns everything published in
# Section 6: unit tests (at build), the three audits + frontier, the
# 15-seed stability sweep, and the solver check on the frozen registers.
# Outputs land in ./out on the host.
set -euo pipefail
cd "$(dirname "$0")"

docker build -f docker/Dockerfile --target highs -t opera-audit:highs .

mkdir -p out
docker run --rm -v "$PWD/out:/opera/results" opera-audit:highs \
    python experiments/binary_classification/run_audit.py
docker run --rm -v "$PWD/out:/opera/results" opera-audit:highs \
    python experiments/binary_classification/seeds_sweep.py
docker run --rm opera-audit:highs \
    python validation/compare_solvers.py highs

echo
echo "Done. JSON outputs: ./out/bcw_audit.json and ./out/bcw_seeds.json"
