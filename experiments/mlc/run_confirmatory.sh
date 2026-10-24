#!/usr/bin/env bash
# Orquestrador das 15 sementes confirmatorias (protocolo v3, passo 4).
# Para cada semente: registro OOF (imagem highs, derivacao SeedSequence)
# + bateria completa (imagem cplex, trajetorias) + hashes no manifesto.
# Reexecutavel: pula o que ja existe.
set -uo pipefail
cd "$(dirname "$0")"
SEEDS="1563301818 2094457352 910208993 2142572456 2355023162 3221619752 2650268534 273224788 4288307182 3586821876 1964561145 2889434148 3699609031 3058505923 1661477577"
mkdir -p confirmatory
MAN=confirmatory/manifest_confirmatory.txt
touch "$MAN"
i=0
for S in $SEEDS; do
  i=$((i+1))
  REG=confirmatory/register_conf_seed${S}.npz
  BAT=confirmatory/battery_conf_seed${S}.json
  echo "=== [$i/15] seed $S $(date '+%H:%M:%S') ==="
  if [ ! -f "$REG" ]; then
    docker run --rm -v "$PWD":/work opera-mlc-highs sh -c \
      "cd /work && python build_register2.py $S 20 && mv register_conf_seed${S}.npz confirmatory/" \
      2>&1 | tail -2 || { echo "FAILED: register for seed $S"; exit 1; }
    sha256sum "$REG" >> "$MAN"
  else
    echo "register already exists, skipping"
  fi
  if [ ! -f "$BAT" ]; then
    docker run --rm -v "$PWD":/work opera-mlc-cplex python /work/run_battery.py \
      --backend cplex --register /work/"$REG" --out /work/"$BAT" \
      --jobs 20 --trajectories 2>&1 | tail -1 \
      || { echo "FAILED: battery for seed $S"; exit 1; }
    sha256sum "$BAT" >> "$MAN"
  else
    echo "battery already exists, skipping"
  fi
done
echo "=== FINISHED $(date '+%H:%M:%S') ==="
wc -l "$MAN"
