"""Deterministic selection of the labels for HiGHS verification in the
scale stratum (protocol, Section 7): if L_elig <= 20, all of them;
otherwise 20 labels: the 5 of lowest prevalence, the 5 of highest, the 5
closest to the median, and 5 of the remaining ones by deterministic hash
(sha256 of "BASE:k"). Uses only the census.
Usage: select_verify_labels.py BASE  -> prints and writes JSON
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np

BASE = sys.argv[1]
HERE = Path(__file__).parent
c = json.loads((HERE / "data_gen" / "censo_elegibilidade.json").read_text())[BASE]
elig = c["elegiveis"]; P = c["prevalencias"]
if len(elig) <= 20:
    sel = sorted(elig)
else:
    order = sorted(elig, key=lambda k: (P[k], k))
    low, high = order[:5], order[-5:]
    med = np.median([P[k] for k in elig])
    by_med = sorted(elig, key=lambda k: (abs(P[k] - med), k))
    mid = [k for k in by_med if k not in low + high][:5]
    rest = [k for k in elig if k not in set(low + high + mid)]
    rest = sorted(rest, key=lambda k: hashlib.sha256(f"{BASE}:{k}".encode()).hexdigest())[:5]
    sel = sorted(low + high + mid + rest)
out = {"base": BASE, "L_eleg": len(elig), "verify_labels": sel}
(HERE / "data_gen" / f"verify_labels_{BASE}.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
