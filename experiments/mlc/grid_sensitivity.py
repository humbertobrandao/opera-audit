"""Experiment A (protocol v3, Section 5): pure economic sensitivity.
M^{0.5} FIXED; payoffs vary on the grid {2,4,8}^2 (9 regimes); 21 labels;
development seed; supplement.
Usage: grid_sensitivity.py --backend {highs,cplex} --register FILE --out OUT.json
"""
import argparse, json, sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from audit_core2 import LabelProblem, costs_int, to_norm

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["highs", "cplex"], required=True)
ap.add_argument("--register", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
LP = LabelProblem
if args.backend == "cplex":
    from backend_cplex import CplexLabelProblem as LP

reg = {k: v for k, v in np.load(args.register, allow_pickle=True).items()}
eligible = reg["eligible"].tolist(); Y = reg["Y"]; n = Y.shape[0]
GRID = [(lm, lp) for lm in (2, 4, 8) for lp in (2, 4, 8)]

out, t0 = {}, time.time()
for i, k in enumerate(eligible):
    yk = Y[:, k].astype(int)
    M05 = (reg[f"probs_{k}"].astype(float) >= 0.5).astype(np.int8)
    P, N = int(yk.sum()), n - int(yk.sum())
    rk = {"P": P, "N": N}
    for lm, lp in GRID:
        Lp = LP(M05, yk, costs_int(P, N, lm, lp))
        s = Lp.solve()
        rk[f"g{lm}{lp}"] = {"U_int": int(s["U"]), "U": to_norm(s["U"], P, N),
                            "verdict": s["verdict"], "t": s["t"],
                            "J": s["J"], "fp": s["fp"], "fn": s["fn"],
                            "seconds": s["seconds"], "nodes": s["nodes"]}
    out[str(k)] = rk
    print(f"[{i+1}/21] k={k} t={time.time()-t0:.0f}s", flush=True)

for lm, lp in GRID:
    go = sum(out[str(k)][f"g{lm}{lp}"]["verdict"] == "go" for k in eligible)
    ub = np.mean([out[str(k)][f"g{lm}{lp}"]["U"] for k in eligible])
    print(f"grid ({lm},{lp}): GO {go}/21  Ubar={ub:7.2f}", flush=True)

json.dump({"meta": {"backend": args.backend, "reading": "m05_fixa",
                    "total_seconds": time.time() - t0}, "labels": out},
          open(args.out, "w"), indent=1)
print("DONE", flush=True)
