#!/usr/bin/env bash
# Pre-certification over ALL the registers already present on the machine.
# Runs no solver: numpy only (opera-mlc-highs image). Re-runnable: skips
# what already exists. Compares against the matching battery when it
# exists next to the register.
#
# Usage: ./run_precert.sh            -> sweeps confirmatory/ and registers_gen/
#        ./run_precert.sh delicious  -> only the registers of that dataset
set -uo pipefail
cd "$(dirname "$0")"
FILTER="${1:-}"
JOBS=20
mkdir -p precert
MAN=precert/manifest_precert.txt
touch "$MAN"

run_one () {  # REG BAT OUT
  REG=$1; BAT=$2; OUT=$3
  [ -f "$OUT" ] && { echo "already exists: $OUT"; return; }
  [ -f "$REG" ] || { echo "missing register: $REG"; return; }
  BATARG=""
  [ -f "$BAT" ] && BATARG="--battery /work/$BAT"
  echo "== precert $(basename "$REG") $(date '+%H:%M:%S')"
  docker run --rm -v "$PWD":/work opera-mlc-highs python \
    /work/precert.py --register /work/"$REG" --out /work/"$OUT" \
    --jobs $JOBS $BATARG 2>&1 | tail -2
  [ -f "$OUT" ] && sha256sum "$OUT" >> "$MAN"
}

# 1) medical (the 15 confirmatory seeds of protocol v3)
if [ -z "$FILTER" ] || [ "$FILTER" = "medical" ]; then
  for REG in confirmatory/register_conf_seed*.npz; do
    [ -e "$REG" ] || continue
    S=$(basename "$REG" .npz); S=${S#register_conf_seed}
    run_one "$REG" "confirmatory/battery_conf_seed${S}.json" \
            "precert/precert_medical_seed${S}.json"
  done
fi

# 2) generality + scale (7x15 + 3x3)
for REG in registers_gen/register_*_seed*.npz; do
  [ -e "$REG" ] || continue
  NAME=$(basename "$REG" .npz); NAME=${NAME#register_}
  B=${NAME%_seed*}; S=${NAME##*_seed}
  if [ -n "$FILTER" ] && [ "$B" != "$FILTER" ]; then continue; fi
  run_one "$REG" "batteries_gen/battery_${B}_seed${S}.json" \
          "precert/precert_${B}_seed${S}.json"
done

echo "===== FINISHED $(date '+%H:%M:%S') ====="
wc -l "$MAN"
