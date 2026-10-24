"""Formulation ablation on a HARD dataset (F1-F4 grid of the revision,
same solver), over the deterministic 20-label sample and the frozen
register. Complements the development-seed ablation (easy dataset,
ablation.py).

Variants (same solver; CPLEX in the study of the paper):
  F1 per instance, uniform constants (big-M = m in both constraints);
  F2 per instance, structural constants (tbar and q_i, as in the body);
  F3 profile-aggregated, structural constants (the register formulation);
  F4 aggregated, indicator constraints (cplex backend only).

Under a reduced time limit (--tl), not every solve proves the optimum:
the identity requirement holds ONLY among variants with optimal status;
for the others, incumbent and status are reported. Metrics per variant:
variables, constraints, T_build, T_solver, nodes, root bound, status.
The model is built SPARSE (the hard dataset makes the dense per-instance
matrix infeasible).

Usage: ablation_hard.py --backend {highs,cplex} --register REG.npz
       --out OUT.json --labels-file verify_labels_X.json
       [--regime R2] [--tl 60] [--readings m05,mtau]
highs: only F1-F3, for functional validation (scipy TL fixed at 600s).
"""
import argparse, json, sys, time
import numpy as np
from pathlib import Path
from scipy import sparse
from scipy.optimize import LinearConstraint, Bounds

sys.path.insert(0, str(Path(__file__).parent))
import audit_core2
from audit_core2 import LabelProblem, costs_int, build_M_tau, REGIMES

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["highs", "cplex"], required=True)
ap.add_argument("--register", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--labels-file", required=True)
ap.add_argument("--regime", default="R2", choices=list(REGIMES))
ap.add_argument("--tl", type=float, default=60.0)
ap.add_argument("--readings", default="m05,mtau")
args = ap.parse_args()

if args.backend == "cplex":
    import backend_cplex
    backend_cplex.TL_SECONDS = float(args.tl)
    BASE = backend_cplex.CplexLabelProblem
else:
    BASE = LabelProblem           # validacao funcional (TL scipy = 600 s)


def make_variant(base, row_level, uniform):
    class Hard(base):
        def __init__(self, M, y, costs):
            super().__init__(M, y, costs)
            if row_level:
                self.Mp = M.copy(); self.yp = y.astype(np.int8)
                self.w = np.ones(len(y), dtype=np.int64)
                self.Pn = M.shape[0]
                self.q = M.sum(axis=1).astype(float)
                r_tp, r_tn, r_fn, r_fp = (int(c) for c in costs)
                self.gain = np.where(self.yp == 1, r_tp - r_fn,
                                     r_fp - r_tn) * self.w
            if uniform:               # uniform big-M in both constraints
                self.q = np.full(self.Pn, float(self.m))

        def _model(self, t_fix=None, card_fix=None, forbid_neg=False,
                   U_floor=None, obj="U", gain=None):
            assert card_fix is None and U_floor is None and not forbid_neg \
                and obj == "U", "ablation variant: main audit only"
            m, Pn, tbar = self.m, self.Pn, float(self.m)
            g = self.gain if gain is None else gain
            nv = m + 1 + Pn
            pidx, jidx = np.nonzero(self.Mp)
            nz = pidx.size
            # rows 2p (activation) and 2p+1 (deactivation), built sparse
            rows = np.concatenate([2 * pidx, 2 * pidx + 1,
                                   np.arange(2 * Pn),
                                   2 * np.arange(Pn), 2 * np.arange(Pn) + 1])
            cols = np.concatenate([jidx, jidx,
                                   np.full(2 * Pn, m),
                                   m + 1 + np.arange(Pn),
                                   m + 1 + np.arange(Pn)])
            data = np.concatenate([np.ones(nz), np.ones(nz),
                                   -np.ones(2 * Pn),
                                   -np.full(Pn, tbar),
                                   -self.q.astype(float)])
            A = sparse.csr_matrix((data, (rows, cols)), shape=(2 * Pn, nv))
            lb = np.empty(2 * Pn); ub = np.empty(2 * Pn)
            lb[0::2] = -tbar;    ub[0::2] = np.inf     # ativacao
            lb[1::2] = -np.inf;  ub[1::2] = -1.0       # desativacao
            c = np.zeros(nv); c[m + 1:] = -g
            lo = np.r_[np.zeros(m), 1.0, np.zeros(Pn)]
            hi = np.r_[np.ones(m), float(m), np.ones(Pn)]
            if t_fix is not None:
                lo[m] = hi[m] = float(t_fix)
            return c, LinearConstraint(A, lb, ub), Bounds(lo, hi), nv
    return Hard


F1 = make_variant(BASE, row_level=True,  uniform=True)
F2 = make_variant(BASE, row_level=True,  uniform=False)
F3 = make_variant(BASE, row_level=False, uniform=False)


def run_explicit(cls, M, y, costs):
    tb0 = time.perf_counter()
    prob = cls(M, y, costs)
    c, cons, bounds, nv = prob._model()
    t_build = time.perf_counter() - tb0
    if args.backend == "cplex":
        res, dt = prob._cplex_solve(c, cons, bounds, nv, False, "benchmark")
        s = prob._extract(res, dt)
        rb = prob._cplex_solve(c, cons, bounds, nv, False, "root")
        root = prob.U0 - rb["bound"]; t_root = rb["seconds"]
    else:
        res, dt = prob._run()
        s = prob._extract(res, dt)
        res2, dt2 = prob._run(relax=True)
        root = prob.U0 - res2.fun; t_root = dt2
    status = getattr(prob, "last_status", "optimal")
    return {"U_int": int(s["U"]), "status": status,
            "optimal": str(status).startswith("optimal"),
            "vars": nv, "cons": int(cons.A.shape[0]),
            "t_build": t_build, "t_solver": s["seconds"],
            "t_root": t_root, "nodes": s["nodes"],
            "root_bound": float(root), "profiles": prob.Pn}


def run_f4(M, y, costs, tl):
    """Aggregated with native indicator constraints (CPLEX only)."""
    import cplex
    prob = F3(M, y, costs)          # aggregation and gains ready
    m, Pn = prob.m, prob.Pn

    def build():
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
            cpx.indicator_constraints.add(
                indvar=f"z{p}", complemented=0,
                lin_expr=cplex.SparsePair(idx, val), sense="G", rhs=0.0)
            cpx.indicator_constraints.add(
                indvar=f"z{p}", complemented=1,
                lin_expr=cplex.SparsePair(idx, val), sense="L", rhs=-1.0)
        cpx.objective.set_sense(cpx.objective.sense.minimize)
        cpx.parameters.threads.set(1)
        cpx.parameters.timelimit.set(tl)
        cpx.parameters.mip.tolerances.mipgap.set(0.0)
        return cpx

    tb0 = time.perf_counter(); cpx = build()
    t_build = time.perf_counter() - tb0
    t0 = time.perf_counter(); cpx.solve(); dt = time.perf_counter() - t0
    stn = cpx.solution.get_status()
    U = prob.U0 - cpx.solution.get_objective_value()
    nodes = cpx.solution.progress.get_num_nodes_processed()
    rbx = build(); rbx.parameters.mip.limits.nodes.set(0)
    t0 = time.perf_counter(); rbx.solve(); t_root = time.perf_counter() - t0
    root = prob.U0 - rbx.solution.MIP.get_best_objective()
    return {"U_int": int(round(U)), "status": str(stn),
            "optimal": stn in (101, 102),
            "vars": m + 1 + Pn, "cons": 2 * Pn,
            "t_build": t_build, "t_solver": dt, "t_root": t_root,
            "nodes": nodes, "root_bound": float(root), "profiles": Pn}


lf = json.load(open(args.labels_file))
labels = lf["verify_labels"]
reg = np.load(args.register, allow_pickle=True)
Y = reg["Y"]; n = Y.shape[0]
lm, lp = REGIMES[args.regime]
readings = args.readings.split(",")

def unpad(fp): return [f[f >= 0] for f in fp]

res, diverg = {}, 0
t_all = time.time()
for i, k in enumerate(labels):
    yk = Y[:, k].astype(int)
    probs = reg[f"probs_{k}"].astype(float)
    folds = unpad(reg[f"folds_{k}"])
    P, N = int(yk.sum()), n - int(yk.sum())
    costs = costs_int(P, N, lm, lp)
    for reading in readings:
        M = ((probs >= 0.5).astype(np.int8) if reading == "m05"
             else build_M_tau(probs, yk, folds, lm, lp))
        entry = {"F1_row_uniforme": run_explicit(F1, M, yk, costs),
                 "F2_row_estrutural": run_explicit(F2, M, yk, costs),
                 "F3_agg_estrutural": run_explicit(F3, M, yk, costs)}
        if args.backend == "cplex":
            entry["F4_agg_indicator"] = run_f4(M, yk, costs, float(args.tl))
        optima = {v: e["U_int"] for v, e in entry.items() if e["optimal"]}
        if len(set(optima.values())) > 1:
            diverg += 1
            print(f"DIVERGENCE among optima k={k} {reading}: {optima}",
                  flush=True)
        res[f"{k}_{reading}"] = entry
        print(f"[{i+1}/{len(labels)}] k={k} {reading} "
              + " ".join(f"{v.split('_')[0]}:"
                         f"{'OPT' if e['optimal'] else 'TL'}"
                         f"/{e['t_solver']:.1f}s" for v, e in entry.items()),
              flush=True)

json.dump({"meta": {"backend": args.backend, "regime": args.regime,
                    "tl": float(args.tl), "labels": labels,
                    "register": Path(args.register).name,
                    "divergencias_entre_otimos": diverg,
                    "total_seconds": time.time() - t_all},
           "cases": res}, open(args.out, "w"), indent=1)
print(f"DONE {time.time()-t_all:.0f}s divergences={diverg} -> {args.out}",
      flush=True)
