"""Measured effect of the precedence inequalities (Proposition on the
precedence-strengthened LP) on the aggregated indicator formulation, over
the same hard sample as the ablation (verify_labels, frozen register,
CPLEX, 1 thread).

Variants:
  F4  aggregated + indicator constraints (identical to ablation_hard.run_f4);
  F5  F4 + precedence inequalities z_i <= z_j over the covering pairs
      (Hasse diagram) of the observed vote profiles, equality z_i = z_j
      between rows with the same vote profile (the internal aggregation
      separates by (profile, y)), and z = 0 on the null profile.

Per cell and variant: U_int, status, optimal?, t_build, t_solver, t_root,
nodes, root_bound, n_prec; and U_mon_int (bounds_hierarchy.bounds_cell)
with the sanity check root_bound(F5) <= U_mon + tol, which validates the
proposition numerically.

Usage: precedence_experiment.py --register REG.npz --out OUT.json
       --labels-file verify_labels_X.json [--regime R2] [--tl 60]
       [--readings m05,mtau]
Requires the CPLEX backend (opera-mlc-cplex image).
"""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import backend_cplex
from audit_core2 import LabelProblem, costs_int, build_M_tau, REGIMES
from bounds_hierarchy import bounds_cell

ap = argparse.ArgumentParser()
ap.add_argument("--register", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--labels-file", required=True)
ap.add_argument("--regime", default="R2", choices=list(REGIMES))
ap.add_argument("--tl", type=float, default=60.0)
ap.add_argument("--readings", default="m05,mtau")
args = ap.parse_args()

import cplex


def covering_pairs(Mp):
    """Pairs (i, j) with profile_i strictly dominated by profile_j and no
    intermediate (transitive reduction), and groups of identical profiles.
    Returns (covering_pairs, equality_pairs, null_rows)."""
    P = np.asarray(Mp, dtype=np.float32)
    V = P.shape[0]
    # dom[i, j] = 1 iff P[i] <= P[j] componentwise (non-strict)
    viol = P @ (1.0 - P).T                     # 0 iff P[i] <= P[j]
    leq = (viol == 0)
    eq = leq & leq.T
    np.fill_diagonal(eq, False)
    strict = leq & ~eq.T & ~np.eye(V, dtype=bool)   # i < j estrito
    # equality: one representative per group
    eq_pairs, seen = [], set()
    for i in range(V):
        if i in seen:
            continue
        grp = [i] + [j for j in np.nonzero(eq[i])[0] if j > i]
        for j in grp[1:]:
            eq_pairs.append((grp[0], int(j))); seen.add(int(j))
    # transitive reduction over the strict relations among representatives
    reps = [i for i in range(V) if i not in seen]
    S = strict[np.ix_(reps, reps)]
    D = S.astype(np.float32)
    has_mid = (D @ D) > 0
    cov = S & ~has_mid
    cov_pairs = [(reps[a], reps[b]) for a, b in zip(*np.nonzero(cov))]
    nulls = [int(i) for i in range(V) if not Mp[i].any()]
    return cov_pairs, eq_pairs, nulls


def build_f4(prob, tl):
    m, Pn = prob.m, prob.Pn
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


def add_precedencia(cpx, cov, eqs, nulls):
    lin, senses, rhs = [], "", []
    for i, j in cov:                       # z_i <= z_j
        lin.append(cplex.SparsePair([f"z{i}", f"z{j}"], [1.0, -1.0]))
        senses += "L"; rhs.append(0.0)
    for i, j in eqs:                       # z_i = z_j (same vote profile)
        lin.append(cplex.SparsePair([f"z{i}", f"z{j}"], [1.0, -1.0]))
        senses += "E"; rhs.append(0.0)
    if lin:
        cpx.linear_constraints.add(lin_expr=lin, senses=senses, rhs=rhs)
    if nulls:
        cpx.variables.set_upper_bounds([(f"z{i}", 0.0) for i in nulls])
    return len(lin) + len(nulls)


def solve_variant(prob, tl, prec):
    tb0 = time.perf_counter()
    cpx = build_f4(prob, tl)
    n_prec = 0
    if prec is not None:
        n_prec = add_precedencia(cpx, *prec)
    t_build = time.perf_counter() - tb0
    t0 = time.perf_counter(); cpx.solve(); dt = time.perf_counter() - t0
    stn = cpx.solution.get_status()
    try:
        U = prob.U0 - cpx.solution.get_objective_value()
        bnd = prob.U0 - cpx.solution.MIP.get_best_objective()
    except cplex.exceptions.CplexError:
        U, bnd = None, None
    nodes = cpx.solution.progress.get_num_nodes_processed()
    # root bound: same model, 0 nodes
    rbx = build_f4(prob, tl)
    if prec is not None:
        add_precedencia(rbx, *prec)
    rbx.parameters.mip.limits.nodes.set(0)
    t0 = time.perf_counter(); rbx.solve(); t_root = time.perf_counter() - t0
    root = prob.U0 - rbx.solution.MIP.get_best_objective()
    return {"U_int": None if U is None else int(round(U)),
            "final_bound": None if bnd is None else float(bnd),
            "status": str(stn), "optimal": stn in (101, 102),
            "t_build": t_build, "t_solver": dt, "t_root": t_root,
            "nodes": nodes, "root_bound": float(root),
            "n_prec": int(n_prec), "profiles": prob.Pn}


def unpad(fp):
    return [f[f >= 0] for f in fp]


lf = json.load(open(args.labels_file))
labels = lf["verify_labels"]
reg = np.load(args.register, allow_pickle=True)
Y = reg["Y"]; n = Y.shape[0]
lm, lp = REGIMES[args.regime]
readings = args.readings.split(",")
backend_cplex.TL_SECONDS = float(args.tl)

res, viol_sanidade = {}, 0
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
        prob = LabelProblem(M, yk, costs)
        prec = covering_pairs(prob.Mp)
        cell = {"P": P, "N": N,
                "F4_agg_indicator": solve_variant(prob, args.tl, None),
                "F5_agg_ind_prec": solve_variant(prob, args.tl, prec),
                "U_mon_int": bounds_cell(M, yk, costs)["U_mon_int"]}
        # sanity (precedence proposition): F5 root <= U_mon, tolerance 1 on the integer scale
        ok = cell["F5_agg_ind_prec"]["root_bound"] <= cell["U_mon_int"] + 1.0
        cell["sanidade_root_le_Umon"] = bool(ok)
        if not ok:
            viol_sanidade += 1
            print(f"[VIOLATION] label {k} {reading}: root(F5)="
                  f"{cell['F5_agg_ind_prec']['root_bound']:.1f} > "
                  f"U_mon={cell['U_mon_int']}", flush=True)
        res[f"{k}_{reading}"] = cell
        print(f"[{i+1}/{len(labels)}] label {k} {reading}: "
              f"F4 {cell['F4_agg_indicator']['status']}"
              f" ({cell['F4_agg_indicator']['t_solver']:.1f}s) | "
              f"F5 {cell['F5_agg_ind_prec']['status']}"
              f" ({cell['F5_agg_ind_prec']['t_solver']:.1f}s, "
              f"{cell['F5_agg_ind_prec']['n_prec']} ineqs.)", flush=True)

out = {"register": args.register, "regime": args.regime, "tl": args.tl,
       "readings": readings, "labels": [int(x) for x in labels],
       "violacoes_sanidade": viol_sanidade,
       "t_total": time.time() - t_all, "cells": res}
Path(args.out).write_text(json.dumps(out, indent=1))
print(f"OK: {len(res)} cells, {viol_sanidade} sanity violations, "
      f"{out['t_total']:.0f}s -> {args.out}")
