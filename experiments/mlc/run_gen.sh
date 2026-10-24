#!/usr/bin/env bash
# Generality + scale orchestrator (frozen protocol sha256 304b2e52...).
# Confirmatory stratum: 7 datasets x 15 seeds. Scale: 3 datasets x 3 seeds.
# Re-runnable (skips what exists). No concurrent heavy load (rule S-8).
#
# Usage: ./run_gen.sh [dataset ...]
#   no arguments  -> runs ALL ten datasets (both strata), as in the paper
#   with datasets -> runs only those (any subset; each dataset keeps its
#                    own stratum: 15 seeds for the confirmatory seven,
#                    3 seeds for the scale giants). Useful to shard the
#                    campaign across machines, one or two datasets each.
#   examples: ./run_gen.sh emotions scene
#             ./run_gen.sh delicious
set -uo pipefail
cd "$(dirname "$0")"
SEEDS15="1563301818 2094457352 910208993 2142572456 2355023162 3221619752 2650268534 273224788 4288307182 3586821876 1964561145 2889434148 3699609031 3058505923 1661477577"
SEEDS3="1563301818 2094457352 910208993"
CONF="emotions scene yeast birds cal500 enron bibtex"
ESCALA="eukaryotego mediamill delicious"
if [ "$#" -gt 0 ]; then
  SEL_CONF=""; SEL_ESC=""
  for A in "$@"; do
    case " $CONF " in *" $A "*) SEL_CONF="$SEL_CONF $A"; continue;; esac
    case " $ESCALA " in *" $A "*) SEL_ESC="$SEL_ESC $A"; continue;; esac
    echo "unknown dataset: $A"
    echo "valid datasets: $CONF $ESCALA"
    exit 1
  done
  CONF="$SEL_CONF"; ESCALA="$SEL_ESC"
fi
echo "selected confirmatory datasets:${CONF:-" (none)"}"
echo "selected scale datasets:${ESCALA:-" (none)"}"
W=20
mkdir -p registers_gen batteries_gen
MAN=batteries_gen/manifest_gen.txt; touch "$MAN"

run_one () {  # base seed trajflag
  B=$1; S=$2; TRAJ=$3
  REG=registers_gen/register_${B}_seed${S}.npz
  BAT=batteries_gen/battery_${B}_seed${S}.json
  if [ -f "registers_gen/register_${B}_seed${S}.FAILED" ]; then
    echo "!! $B seed $S: TECHNICAL FAILURE recorded, skipping"; return
  fi
  if [ ! -f "$REG" ]; then
    echo "== register $B seed $S $(date '+%H:%M:%S')"
    docker run --rm -v "$PWD":/work opera-mlc-highs python \
      /work/build_register_gen.py "$B" "$S" $W 2>&1 | tail -2 || return
    [ -f "$REG" ] && sha256sum "$REG" >> "$MAN"
  fi
  if [ -f "$REG" ] && [ ! -f "$BAT" ]; then
    echo "== battery $B seed $S $(date '+%H:%M:%S')"
    docker run --rm -v "$PWD":/work opera-mlc-cplex python \
      /work/run_battery_gen.py --base "$B" --backend cplex \
      --register /work/"$REG" --out /work/"$BAT" --jobs $W $TRAJ 2>&1 | tail -1
    [ -f "$BAT" ] && sha256sum "$BAT" >> "$MAN"
  fi
}

echo "===== CONFIRMATORY STRATUM ====="
for B in $CONF; do for S in $SEEDS15; do run_one "$B" "$S" "--trajectories"; done; done
echo "===== SCALE STRATUM ====="
for B in $ESCALA; do for S in $SEEDS3; do run_one "$B" "$S" "--trajectories"; done; done
echo "===== HiGHS VERIFICATION (seed 1) ====="
S1=1563301818
for B in $CONF; do
  V=batteries_gen/verify_${B}_seed${S1}.json
  [ -f "$V" ] && continue
  docker run --rm -v "$PWD":/work opera-mlc-highs python \
    /work/run_battery_gen.py --base "$B" --backend highs --core \
    --register /work/registers_gen/register_${B}_seed${S1}.npz \
    --out /work/"$V" --jobs $W 2>&1 | tail -1
done
for B in $ESCALA; do
  docker run --rm -v "$PWD":/work opera-mlc-highs python \
    /work/select_verify_labels.py "$B"
done
echo "(reduced verification of the giants: run after inspecting verify_labels_*.json; see README)"
echo "===== FINISHED $(date '+%H:%M:%S') ====="
wc -l "$MAN"
