"""Formulation ablation (protocol v3, Section 11) - development seed, R1,
both readouts, 21 labels.

Four frozen variants:
  V1 explicit row-level (no profile aggregation);
  V2 explicit profile-aggregated (the register formulation);
  V3 aggregated with exact enumeration of t (root bound = max_t UB_root(t));
  V4 aggregated with indicator constraints (CPLEX only).
Requires identical U_int across the variants. Metrics: variables,
constraints, T_build, T_solver, T_end_to_end, nodes, root bound.
Usage: ablation.py --backend {highs,cplex} --register REG --out OUT.json
"""
import argparse, json, sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from audit_core2 import LabelProblem, costs_int, to_norm, build_M_tau

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["highs", "cplex"], required=True)
ap.add_argument("--register", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

if args.backend == "cplex":
    from backend_cplex import CplexLabelProblem as BASE
else:
    BASE = LabelProblem


def make_rowlevel(base):
    class RowLevel(base):
        def __init__(self, M, y, costs):
            super().__init__(M, y, costs)
            self.Mp = M.copy(); self.yp = y.astype(np.int8)
            self.w = np.ones(len(y), dtype=np.int64)
            self.Pn = M.shape[0]; self.q = M.sum(axis=1).astype(float)
            r_tp, r_tn, r_fn, r_fp = (int(c) for c in costs)
            self.gain = np.where(self.yp == 1, r_tp - r_fn, r_fp - r_tn) * self.w
    return RowLevel


ROW = make_rowlevel(BASE)


def sizes(prob):
    c, cons, bounds, nv = prob._model()
    return nv, cons.A.shape[0]


def run_v12(prob):
    tb0 = time.perf_counter(); nv, nc = sizes(prob); tb = time.perf_counter() - tb0
    s = prob.solve()
    rb = prob.root_bound()
    return {"U_int": int(s["U"]), "vars": nv, "cons": nc,
            "t_solver": s["seconds"], "nodes": s["nodes"],
            "root_bound": float(rb["root_bound"])}


def run_v3(prob):
    """Exact enumeration of t: one solve with t fixed per value of t."""
    bestU, tot_t, tot_nodes, rbs = None, 0.0, 0, []
    for t in range(1, prob.m + 1):
        res, dt = prob._run(t_fix=t)
        s = prob._extract(res, dt)
        tot_t += dt; tot_nodes += s["nodes"]
        if bestU is None or s["U"] > bestU:
            bestU = s["U"]
        if args.backend == "cplex":
            c, cons, bounds, nv = prob._model(t_fix=t)
            out = prob._cplex_solve(c, cons, bounds, nv, False, "root")
            rbs.append(prob.U0 - out["bound"]); tot_t += out["seconds"]
        else:
            res2, dt2 = prob._run(relax=True, t_fix=t)
            rbs.append(prob.U0 - res2.fun); tot_t += dt2
    nv, nc = sizes(prob)
    return {"U_int": int(bestU), "vars": nv - 1, "cons": nc,
            "t_solver": tot_t, "nodes": tot_nodes,
            "root_bound": float(max(rbs))}


def run_v4(prob):
    """Native CPLEX indicator constraints (cplex backend only)."""
    import cplex
    m, Pn = prob.m, prob.Pn
    tb0 = time.perf_counter()
    cpx = cplex.Cplex()
    for st in (cpx.set_log_stream, cpx.set_results_stream,
               cpx.set_warning_stream, cpx.set_error_stream):
        st(None)
    names_x = [f"x{j}" for j in range(m)]
    cpx.variables.add(obj=[0.0] * m, lb=[0] * m, ub=[1] * m,
                      types="B" * m, names=names_x)
    cpx.variables.add(obj=[0.0], lb=[1], ub=[m], types="I", names=["t"])
    cpx.variables.add(obj=[-float(g) for g in prob.gain], lb=[0] * Pn,
                      ub=[1] * Pn, types="B" * Pn,
                      names=[f"z{p}" for p in range(Pn)])
    for p in range(Pn):
        idx = [names_x[j] for j in range(m) if prob.Mp[p, j]] + ["t"]
        val = [1.0] * (len(idx) - 1) + [-1.0]
        cpx.indicator_constraints.add(indvar=f"z{p}", complemented=0,
                                      lin_expr=cplex.SparsePair(idx, val),
                                      sense="G", rhs=0.0)
        cpx.indicator_constraints.add(indvar=f"z{p}", complemented=1,
                                      lin_expr=cplex.SparsePair(idx, val),
                                      sense="L", rhs=-1.0)
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    cpx.parameters.threads.set(1)
    cpx.parameters.timelimit.set(600.0)
    cpx.parameters.mip.tolerances.mipgap.set(0.0)
    tb = time.perf_counter() - tb0
    t0 = time.perf_counter(); cpx.solve(); dt = time.perf_counter() - t0
    U = prob.U0 - cpx.solution.get_objective_value()
    nodes = cpx.solution.progress.get_num_nodes_processed()
    return {"U_int": int(round(U)), "vars": m + 1 + Pn,
            "cons": 2 * Pn, "t_solver": dt, "nodes": nodes,
            "root_bound": None}


reg = {k: v for k, v in np.load(args.register, allow_pickle=True).items()}
eligible = reg["eligible"].tolist(); Y = reg["Y"]; n = Y.shape[0]

def unpad(fp): return [f[f >= 0] for f in fp]

res, bad = {}, 0
t_all = time.time()
for i, k in enumerate(eligible):
    yk = Y[:, k].astype(int)
    probs = reg[f"probs_{k}"].astype(float)
    folds = unpad(reg[f"folds_{k}"])
    P, N = int(yk.sum()), n - int(yk.sum())
    costs = costs_int(P, N, 2, 2)                     # R1
    for reading, M in (("m05", (probs >= 0.5).astype(np.int8)),
                       ("mtau", build_M_tau(probs, yk, folds, 2, 2))):
        tb0 = time.perf_counter()
        agg = BASE(M, yk, costs)
        t_build_agg = time.perf_counter() - tb0
        tb0 = time.perf_counter()
        row = ROW(M, yk, costs)
        t_build_row = time.perf_counter() - tb0
        v1 = run_v12(row); v1["t_build"] = t_build_row
        v2 = run_v12(agg); v2["t_build"] = t_build_agg
        v3 = run_v3(agg);  v3["t_build"] = t_build_agg
        entry = {"V1_row": v1, "V2_agg": v2, "V3_enum_t": v3,
                 "profiles": agg.Pn, "compress": n / agg.Pn}
        if args.backend == "cplex":
            v4 = run_v4(agg); v4["t_build"] = t_build_agg
            entry["V4_indicator"] = v4
        Us = {kk: v["U_int"] for kk, v in entry.items() if kk.startswith("V")}
        if len(set(Us.values())) != 1:
            bad += 1
            print(f"DIVERGENCE k={k} {reading}: {Us}", flush=True)
        for v in entry:
            if v.startswith("V"):
                entry[v]["t_end_to_end"] = entry[v]["t_build"] + entry[v]["t_solver"]
        res[f"{k}_{reading}"] = entry
    print(f"[{i+1}/21] k={k} ok t={time.time()-t_all:.0f}s", flush=True)

json.dump({"meta": {"backend": args.backend, "regime": "R1",
                    "total_seconds": time.time() - t_all,
                    "divergencias": bad}, "cases": res},
          open(args.out, "w"), indent=1)
print(f"DONE {time.time()-t_all:.0f}s divergences={bad} -> {args.out}",
      flush=True)
