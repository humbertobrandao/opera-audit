"""Sanity test of the pre-certification against exhaustive enumeration.

For small random instances (m <= 7), enumerates ALL the decisions (S, t)
and checks: best_cand <= U* <= UB; go_pre => U* > 0; nogo_pre => U* <= 0;
identical U0. Uses the real routines of precert.py.
"""
import itertools, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "opera-audit/experiments/mlc"))
sys.path.insert(0, str(HERE))
from audit_core2 import costs_int
from precert import precert_cell


def brute_force(M, y, costs):
    n, m = M.shape
    r_tp, r_tn, r_fn, r_fp = (int(c) for c in costs)
    P, N = int(y.sum()), int((1 - y).sum())
    U0 = r_tn * N + r_fn * P
    best = U0                                 # null committee
    for sz in range(1, m + 1):
        for S in itertools.combinations(range(m), sz):
            sc = M[:, list(S)].sum(axis=1)
            for t in range(1, sz + 1):
                act = sc >= t
                tp = int(y[act].sum()); fp = int((1 - y[act]).sum())
                U = (r_tp * tp + r_fn * (P - tp)
                     + r_fp * fp + r_tn * (N - fp))
                best = max(best, U)
    return best, U0


rng = np.random.default_rng(20260812)
bad, closed_go, closed_nogo, total = 0, 0, 0, 0
for trial in range(400):
    n = int(rng.integers(6, 30))
    m = int(rng.integers(2, 8))
    dens = rng.uniform(0.1, 0.9)
    M = (rng.random((n, m)) < dens).astype(np.int8)
    y = (rng.random(n) < rng.uniform(0.1, 0.9)).astype(int)
    if y.sum() == 0 or y.sum() == n:
        continue
    lm, lp = [(2, 2), (4, 2), (2, 4)][trial % 3]
    P, N = int(y.sum()), n - int(y.sum())
    costs = costs_int(P, N, lm, lp)
    out = precert_cell(M, y, costs)
    Ustar, U0 = brute_force(M, y, costs)
    total += 1
    bc = out["best_cand"]["U_int"]
    ub = out["ub_simple_int"]
    ok = (bc <= Ustar) and (Ustar <= ub)
    if out["pre_verdict"] == "go_pre":
        ok = ok and Ustar > 0
        closed_go += 1
    if out["pre_verdict"] == "nogo_pre":
        ok = ok and Ustar <= 0
        closed_nogo += 1
    assert out["U0_int"] == U0, f"divergent U0 at trial {trial}"
    if not ok:
        bad += 1
        print(f"FAIL trial={trial} best={bc} U*={Ustar} UB={ub} "
              f"verdict={out['pre_verdict']}")
print(f"{'OK' if bad == 0 else 'FAILED'}: {total} instances, "
      f"{bad} violations; the pre-test closed go={closed_go} "
      f"nogo={closed_nogo} ({100*(closed_go+closed_nogo)/total:.0f}%)")
