#!/usr/bin/env bash
# Cross-check track (requires YOUR IBM CPLEX Studio installer).
# Expects the Linux x86-64 installer at docker/cplex_installer.bin
# (CPLEX Studio 22.1.1; a later version may work but the published
# numbers were certified with 22.1.1).
#
# The image built here contains proprietary IBM software.
# NEVER push it to any registry. Keep it local (docker save/load).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f docker/cplex_installer.bin ]; then
    echo "ERROR: docker/cplex_installer.bin not found."
    echo "Copy your IBM CPLEX Studio 22.1.1 Linux x86-64 installer there, e.g.:"
    echo "  cp ~/Downloads/cplex_studio2211.linux_x86_64.bin docker/cplex_installer.bin"
    exit 1
fi

docker build -f docker/Dockerfile --target cplex -t opera-audit:cplex .

# Re-solve the exact registers of the paper with BOTH backends and assert
# that every published optimum (U*, FN, FP) is certified.
docker run --rm opera-audit:cplex python validation/compare_solvers.py

echo
echo "Optional: keep the image locally (do NOT publish it):"
echo "  docker save opera-audit:cplex | gzip > opera-audit-cplex.tar.gz"
