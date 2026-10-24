#!/usr/bin/env bash
# F1-F4 ablation on the hard dataset (Mediamill, deterministic sample of
# 20 labels, R2, both readouts, CPLEX, TL 60 s per solve). Re-runnable.
# Estimate: <= ~6 h (worst case; dominated by the TL solves and by the
# sparse build of the per-instance variants). Not a timing measurement of
# the paper: a RELATIVE comparison between formulations on the same solver.
set -uo pipefail
cd "$(dirname "$0")"
REG=registers_gen/register_mediamill_seed1563301818.npz
LAB=data_gen/verify_labels_mediamill.json
OUT=ablation_hard_mediamill_R2.json
[ -f "$OUT" ] && { echo "already exists: $OUT"; exit 0; }
[ -f "$REG" ] || { echo "missing register: $REG"; exit 1; }
[ -f "$LAB" ] || { echo "missing sample: $LAB"; exit 1; }
docker run --rm -v "$PWD":/work opera-mlc-cplex python \
  /work/ablation_hard.py --backend cplex \
  --register /work/"$REG" --out /work/"$OUT" \
  --labels-file /work/"$LAB" --regime R2 --tl 60
sha256sum "$OUT"
