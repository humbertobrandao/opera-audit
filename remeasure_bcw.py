"""CLEAN re-measurement of the binary-study solves (Section 6.2) over the
FROZEN registers of the repository (results/registers_seed20260811.npz).

Trains nothing: loads the published matrices M and redoes only the three
solves reported in the paper (pools 3, 6 and 4), checking that the optima
reproduce the published values exactly before accepting the times.

Usage (at the root of the opera-audit clone):
    python remeasure_bcw.py
Output: results/bcw_times_<hostname>.json
"""
from __future__ import annotations
import json, socket, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from opera.model import Instance, solve  # noqa: E402

PUBLICADOS = {3: -81_032, 6: -28_771, 4: 21_477}
POOL_LABELS = {3: "initial pool (m=5, 0.5 readout)",
           6: "enlarged pool (m=25, 0.5 readout)",
           4: "rebuilt pool (m=5, cost-derived readout)"}

reg = np.load(ROOT / "results" / "registers_seed20260811.npz",
              allow_pickle=False)
y = reg["y"].astype(int)
costs = reg["costs"].astype(float)

out = {"hostname": socket.gethostname(),
       "register": "registers_seed20260811.npz (frozen)",
       "backend": "highs", "audits": []}
ok = True
for idx in (3, 6, 4):
    M = reg[f"M_{idx}"].astype(np.int8)
    t0 = time.perf_counter()
    sol = solve(Instance(M, y, costs[None, :]), backend="highs")
    wall = time.perf_counter() - t0
    match = int(round(sol.U_star)) == PUBLICADOS[idx]
    ok &= match
    print(f"pool {idx} ({POOL_LABELS[idx]}): U*={sol.U_star:.0f} "
          f"{'== published OK' if match else '!= PUBLISHED ' + str(PUBLICADOS[idx])} "
          f"| solve={sol.seconds:.4f}s | total wall={wall:.4f}s")
    out["audits"].append(dict(pool=idx, label=POOL_LABELS[idx],
                              U_star=float(sol.U_star),
                              published=PUBLICADOS[idx], match=bool(match),
                              solver_seconds=float(sol.seconds),
                              wall_seconds=float(wall)))

name = ROOT / "results" / f"bcw_times_{socket.gethostname()}.json"
json.dump(out, open(name, "w"), indent=2)
print(("ALL optima reproduced; times are valid." if ok else
       "Optimum DIVERGENCE: do NOT use the times.") + f" -> {name}")
sys.exit(0 if ok else 1)
