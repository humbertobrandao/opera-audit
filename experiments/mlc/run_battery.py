"""Full battery of the MLC protocol v3 over an OOF register.

Usage: run_battery.py --backend {highs,cplex} --register REG.npz --out OUT.json
       [--trajectories] [--jobs N]

Per label x {R1,R2,R3} x {m05, mtau}: joint audit (benchmark), root bound,
t=1 (G_cons), exact min|J|, A-OR, A-MAJ, B; full frontier on the tau*
readouts (R1-R3). With --trajectories (cplex only): an additional
instrumented run of the main audit (T_verdict, T_Delta). R4 by composition.
"""
import argparse, json, sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from audit_core2 import (LabelProblem, costs_int, to_norm, build_M_tau,
                         REGIMES, R4_PROFILES)

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["highs", "cplex"], required=True)
ap.add_argument("--register", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--trajectories", action="store_true")
ap.add_argument("--jobs", type=int, default=1)
args = ap.parse_args()

if args.backend == "cplex":
    from backend_cplex import CplexLabelProblem as LP
else:
    LP = LabelProblem

_reg = np.load(args.register, allow_pickle=True)
reg = {k: _reg[k] for k in _reg.files}      # materialize: picklable for workers
eligible = reg["eligible"].tolist(); Y = reg["Y"]; n = Y.shape[0]

def unpad(fp): return [f[f >= 0] for f in fp]

R4_SEED = 20260812
rng = np.random.default_rng(R4_SEED)
L = len(eligible)
base = int(np.ceil(L / 3))
prof_idx = np.array(sum([[i] * base for i in range(3)], [])[:L])
rng.shuffle(prof_idx)
r4_assign = {k: R4_PROFILES[prof_idx[i]] for i, k in enumerate(eligible)}

def one_label(k):
    yk = Y[:, k].astype(int)
    probs = reg[f"probs_{k}"].astype(float)
    folds = unpad(reg[f"folds_{k}"])
    P, N = int(yk.sum()), n - int(yk.sum())
    rk = {"P": P, "N": N}
    M05 = (probs >= 0.5).astype(np.int8)
    for rname, (lm, lp) in REGIMES.items():
        costs = costs_int(P, N, lm, lp)
        Mtau = build_M_tau(probs, yk, folds, lm, lp)
        for reading, M in (("m05", M05), ("mtau", Mtau)):
            tb0 = time.perf_counter()
            Lp = LP(M, yk, costs)
            t_build = time.perf_counter() - tb0
            s = Lp.solve()
            rb = Lp.root_bound()
            s1 = Lp.solve_t1()
            sc = Lp.solve_min_card(s["U"])
            bA1 = Lp.baseline_A_OR(); bA2 = Lp.baseline_A_MAJ(); bB = Lp.baseline_B()
            d = {"U_int": int(s["U"]), "U": to_norm(s["U"], P, N),
                 "U0_int": int(s["U0"]), "verdict": s["verdict"],
                 "dU": to_norm(s["dU"], P, N), "t": s["t"],
                 "J_desc": s["J"], "Jmin": int(sc["Jmin"]),
                 "fp": s["fp"], "fn": s["fn"],
                 "gcons_int": int(s["U"] - s1["U"]),
                 "gcons": to_norm(s["U"] - s1["U"], P, N),
                 "gap_AOR_int": int(s["U"] - bA1["U"]),
                 "gap_AMAJ_int": int(s["U"] - bA2["U"]),
                 "gap_B_int": int(s["U"] - bB["U"]),
                 "root_bound": float(rb["root_bound"]),
                 "root_gap_abs": to_norm(rb["root_bound"] - s["U"], P, N),
                 "profiles": s["profiles"], "compress": n / max(s["profiles"], 1),
                 "t_build": t_build, "t_solver": s["seconds"],
                 "t_end_to_end": t_build + s["seconds"],
                 "nodes": s["nodes"],
                 "status": getattr(Lp, "last_status", "optimal")}
            if args.trajectories and args.backend == "cplex":
                si = Lp.solve_instrumented()
                d["T_verdict"] = si["T_verdict"]; d["T_delta"] = si["T_delta"]
                d["traj_points"] = si["traj_points"]
            if reading == "mtau":
                d["frontier"] = {kk: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                 for kk, v in Lp.frontier(s["U"]).items()}
            rk[f"{rname}_{reading}"] = d
    lm, lp = r4_assign[k]
    src = {(2, 2): "R1", (4, 2): "R2", (2, 4): "R3"}[(int(lm), int(lp))]
    rk["R4_profile"] = f"({int(lm)},{int(lp)})"; rk["R4_source"] = src
    return str(k), rk

t_all = time.time()
if args.jobs > 1:
    from joblib import Parallel, delayed
    pairs = Parallel(n_jobs=args.jobs)(delayed(one_label)(k) for k in eligible)
else:
    pairs = []
    for i, k in enumerate(eligible):
        pairs.append(one_label(k))
        r = pairs[-1][1]
        print(f"[{i+1}/{L}] k={k} R1tau U={r['R1_mtau']['U']:7.2f} "
              f"{r['R1_mtau']['verdict']:5s} t={time.time()-t_all:.0f}s",
              flush=True)

out = {"meta": {"backend": args.backend, "register": Path(args.register).name,
                "r4_seed": R4_SEED, "trajectories": bool(args.trajectories),
                "total_seconds": time.time() - t_all},
       "labels": dict(pairs)}
json.dump(out, open(args.out, "w"), indent=1)
print(f"DONE {out['meta']['total_seconds']:.0f}s -> {args.out}", flush=True)
