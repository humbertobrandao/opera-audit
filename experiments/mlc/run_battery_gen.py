"""Multi-dataset battery: identical to run_battery.py, with R4 per dataset
(entropy SeedSequence([20260812, dataset_index]) over the frozen
alphabetical order) and groups of size +-1 when L_elig is not a multiple
of 3.

Usage: run_battery_gen.py --base B --backend {highs,cplex} --register REG
       --out OUT.json [--trajectories] [--jobs N]
"""
import argparse, json, sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from audit_core2 import (LabelProblem, costs_int, to_norm, build_M_tau,
                         REGIMES, R4_PROFILES)

BASES_ORDEM = ["bibtex", "birds", "cal500", "delicious", "emotions",
               "enron", "eukaryotego", "mediamill", "scene", "yeast"]

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, choices=BASES_ORDEM)
ap.add_argument("--backend", choices=["highs", "cplex"], required=True)
ap.add_argument("--register", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--trajectories", action="store_true")
ap.add_argument("--jobs", type=int, default=1)
ap.add_argument("--labels", default=None,
                help="subset of labels (comma-separated ids), for the reduced verification")
ap.add_argument("--core", action="store_true",
                help="core-certificate verification mode: only the main "
                     "audit and t=1 (U_int, verdict, gcons_int); no min|J|, "
                     "baselines, frontier or root bound")
args = ap.parse_args()

if args.backend == "cplex":
    from backend_cplex import CplexLabelProblem as LP
else:
    LP = LabelProblem

# LAZY read per worker: each task opens the npz and reads only the arrays
# of its own label (~MBs), instead of serializing the whole register to
# every worker (delicious has ~2.6 GB and used to break the joblib
# dispatch). Nothing numeric changes: same arrays, same solve order.
REG_PATH = args.register
_reg = np.load(REG_PATH, allow_pickle=True)
eligible = _reg["eligible"].tolist(); n = _reg["Y"].shape[0]
del _reg
subset = None
if args.labels:
    subset = sorted(int(x) for x in args.labels.split(","))
    assert set(subset) <= set(eligible), "label outside the eligible set"

def unpad(fp): return [f[f >= 0] for f in fp]

# R4 per dataset: derived entropy, balanced groups +-1
r4_entropy = int(np.random.SeedSequence(
    [20260812, BASES_ORDEM.index(args.base)]).generate_state(1)[0])
rng = np.random.default_rng(r4_entropy)
L = len(eligible)
base_ct = L // 3
rem = L - 3 * base_ct
sizes = [base_ct + (1 if i < rem else 0) for i in range(3)]
prof_idx = np.array(sum([[i] * s for i, s in enumerate(sizes)], []))
rng.shuffle(prof_idx)
r4_assign = {k: R4_PROFILES[prof_idx[i]] for i, k in enumerate(eligible)}


def one_label(k):
    reg = np.load(REG_PATH, allow_pickle=True)   # lazy: reads only what it needs
    yk = reg["Y"][:, k].astype(int)
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
            s1 = Lp.solve_t1()
            if args.core:
                # core certificates only (cross-solver verification):
                # both solves have a trivial incumbent, so HiGHS is
                # never left without a solution within the frozen TL.
                rk[f"{rname}_{reading}"] = {
                    "U_int": int(s["U"]), "U": to_norm(s["U"], P, N),
                    "U0_int": int(s["U0"]), "verdict": s["verdict"],
                    "dU": to_norm(s["dU"], P, N), "t": s["t"],
                    "fp": s["fp"], "fn": s["fn"],
                    "gcons_int": int(s["U"] - s1["U"]),
                    "gcons": to_norm(s["U"] - s1["U"], P, N),
                    "profiles": s["profiles"],
                    "t_build": t_build, "t_solver": s["seconds"],
                    "t_end_to_end": t_build + s["seconds"],
                    "status": getattr(Lp, "last_status", "optimal")}
                continue
            rb = Lp.root_bound()
            sc = Lp.solve_min_card(s["U"])
            bA1 = Lp.baseline_A_OR(); bA2 = Lp.baseline_A_MAJ(); bB = Lp.baseline_B()
            dd = {"U_int": int(s["U"]), "U": to_norm(s["U"], P, N),
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
                dd["T_verdict"] = si["T_verdict"]; dd["T_delta"] = si["T_delta"]
                dd["traj_points"] = si["traj_points"]
            if reading == "mtau":
                dd["frontier"] = {kk: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                  for kk, v in Lp.frontier(s["U"]).items()}
            rk[f"{rname}_{reading}"] = dd
    lm, lp = r4_assign[k]
    src = {(2, 2): "R1", (4, 2): "R2", (2, 4): "R3"}[(int(lm), int(lp))]
    rk["R4_profile"] = f"({int(lm)},{int(lp)})"; rk["R4_source"] = src
    return str(k), rk

t_all = time.time()
targets = subset if subset is not None else eligible
if args.jobs > 1:
    from joblib import Parallel, delayed
    pairs = Parallel(n_jobs=args.jobs)(delayed(one_label)(k) for k in targets)
else:
    pairs = []
    for i, k in enumerate(targets):
        pairs.append(one_label(k))
        r = pairs[-1][1]
        print(f"[{i+1}/{L}] k={k} R1tau U={r['R1_mtau']['U']:7.2f} "
              f"{r['R1_mtau']['verdict']:5s} t={time.time()-t_all:.0f}s", flush=True)

out = {"meta": {"base": args.base, "backend": args.backend, "core": bool(args.core),
                "register": Path(args.register).name,
                "r4_entropy": r4_entropy, "trajectories": bool(args.trajectories),
                "total_seconds": time.time() - t_all},
       "labels": dict(pairs)}
json.dump(out, open(args.out, "w"), indent=1)
print(f"DONE {out['meta']['total_seconds']:.0f}s -> {args.out}", flush=True)
