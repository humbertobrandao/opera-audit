"""
Single-label case study of the OPERA paper — the audits of Section 6.

Runs the full pre-registered grid of six pools over one history and one cost
structure. The paper reports three of them, in this order:

    pool 3  initial       m = 5   well-fitted networks, threshold 0.5   -> no-go
    pool 6  enlarged      m = 25  well-fitted networks, threshold 0.5   -> no-go
    pool 4  rebuilt       m = 5   same five networks, cost threshold    -> go

The grid is crossed — fit quality against how each classifier is read — so
that every cell isolates one hypothesis about what would change the verdict:
the quality of the fit, the reading of the classifier, or the size of the pool.

Cost structure — two conditions, both asserted before anything is audited:

  (D)  |r_fn| > r_tp|I+| + r_tn|I-|        false-negative dominance
       Every decision with FN >= 1 has negative utility. This is how the paper
       expresses "a missed malignancy is not compensable" without an infinite
       parameter.

  (T)  r_tp|I+| + r_fp|I-| <= 0            trivial policy unprofitable
       Without (T), a *go* verdict would be reachable by referring the whole
       history for biopsy, and the experiment would demonstrate nothing. (T) is
       a requirement of honest design, not of the model.

Usage:
    python run_audit.py              # HiGHS (default; the paper's solver of record)
    python run_audit.py cplex        # IBM CPLEX cross-check (needs the cplex wheel)

Output: results/bcw_audit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from opera.model import Instance, solve                              # noqa: E402
from pools_mlp import (SEED, POOLS, OUTER_FOLDS, load, record,        # noqa: E402
                       cost_structure)

BACKEND = "highs"


def main(backend: str = BACKEND, seed: int = SEED):
    X, yh = load(ROOT / "data" / "breast-cancer-wisconsin.data")
    npos, nneg = int(yh.sum()), int((1 - yh).sum())
    costs, dom, triv = cost_structure(npos, nneg)
    r_tp, r_tn, r_fn, r_fp = costs

    print(f"seed {seed} | audited history: ALL {len(yh)} records "
          f"(|I+|={npos}, |I-|={nneg}) | register: {OUTER_FOLDS}-fold out-of-fold")
    print(f"(D) |r_fn| > {dom:.0f}   -> r_fn = {r_fn:.0f}")
    print(f"(T) |r_fp| >= {triv:.2f}  -> r_fp = {r_fp:.0f}")
    U0 = r_tn * nneg + r_fn * npos
    print(f"U0 (flag nobody)   = {U0:.0f}")
    print(f"U  (flag everyone) = {r_tp*npos + r_fp*nneg:.0f}"
          f"   <- unprofitable, as (T) requires\n")

    rows = []
    for label, builder, by_cost in POOLS:
        M, names, taus = record(builder(seed), X, yh,
                                costs=costs if by_cost else None, seed=seed)
        m = M.shape[1]
        acc = [float((M[:, j] == yh).mean()) for j in range(m)]
        # FN of the disjunction of every member: the most permissive decision
        # available. If it is positive, no decision in the pool covers the
        # history, and no rearrangement of the consensus can recover it.
        fn_union = int((((M.sum(axis=1) >= 1) == 0) & (yh == 1)).sum())

        sol = solve(Instance(M, yh, costs[None, :]), backend=backend)
        cm = sol.confusion[0]
        sel = [names[j] for j in sol.selected[0]]

        print(f"=== {label} ===")
        print(f"  individual accuracy: min={min(acc):.4f} max={max(acc):.4f}   "
              f"FN of the union (t=1, all) = {fn_union}")
        print(f"  U0={sol.U0:.0f}  U*={sol.U_star:.0f}  dU*={sol.delta:.0f}"
              f"  ->  {sol.verdict.upper()}")
        print(f"  d*: |J|={len(sel)} t={sol.thresholds[0]}  "
              f"TP={cm['tp']} FN={cm['fn']} TN={cm['tn']} FP={cm['fp']}   "
              f"({sol.seconds:.3f}s, {sol.backend})\n")

        rows.append(dict(pool=label, m=m, U0=sol.U0, U_star=sol.U_star,
                         delta=sol.delta, verdict=sol.verdict, J=sel,
                         t=sol.thresholds[0], taus=[round(t, 4) for t in taus],
                         fn_union=fn_union, acc_min=min(acc), acc_max=max(acc),
                         seconds=round(sol.seconds, 3), backend=sol.backend, **cm))

    # Critical price of an unnecessary referral in the rebuilt pool.
    # Under (D) the optimum has FN = 0, so U = r_tp|I+| + r_tn(|I-|-FP) + r_fp·FP
    # and  go  <=>  |r_fp| < (r_tp|I+| + r_tn(|I-|-FP)) / FP.
    row = next(x for x in rows if x["pool"].startswith("4."))
    frontier = None
    if row["verdict"] == "go" and row["fn"] == 0 and row["fp"] > 0:
        rho = (r_tp * npos + r_tn * (nneg - row["fp"])) / row["fp"]
        frontier = dict(fp=row["fp"], critical_price=float(rho),
                        factor=float(rho / abs(r_fp)))
        print(f"critical price of an unnecessary referral in the rebuilt pool: "
              f"|r_fp|* = {rho:.1f}  ({rho/abs(r_fp):.1f}x the adopted value)")

    res = dict(seed=seed, backend=backend, n_history=len(yh),
               outer_folds=OUTER_FOLDS,
               npos=npos, nneg=nneg,
               costs=dict(r_tp=r_tp, r_tn=r_tn, r_fn=r_fn, r_fp=r_fp),
               dominance_threshold=dom, trivial_threshold=triv,
               U0=U0, U_trivial=r_tp * npos + r_fp * nneg,
               frontier=frontier, audits=rows)
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    (out / "bcw_audit.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nwritten to {out / 'bcw_audit.json'}")
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else BACKEND)
