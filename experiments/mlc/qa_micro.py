"""Fixed QA battery: micro-instances with exhaustive enumeration.
m<=8, n<=20, several payoffs and densities. Requires U_enum == U_milp
(exact, integer scale) for: the joint audit, restricted t=1, min|J| and
FP=0. Battery seed: 20260815 (fixed; the collection is deterministic).
"""
import sys, itertools, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from audit_core2 import LabelProblem, costs_int

if "--backend" in sys.argv and sys.argv[sys.argv.index("--backend") + 1] == "cplex":
    from backend_cplex import CplexLabelProblem as LabelProblem  # noqa: F811
    print("QA with the CPLEX backend")

QA_SEED = 20260815
rng = np.random.default_rng(QA_SEED)

def enum_all(M, y, costs):
    """Brute force: every subset x, t in {1..m}."""
    n, m = M.shape
    r_tp, r_tn, r_fn, r_fp = (int(c) for c in costs)
    P, N = int(y.sum()), int(n - y.sum())
    U0 = r_tn * N + r_fn * P
    best_U, best_t1, best_card = U0, U0, (U0, 0)
    best_fp0 = U0 if True else None
    # FP=0: best U among the solutions without FP
    best_fp0 = U0
    for bits in itertools.product([0, 1], repeat=m):
        x = np.array(bits)
        S = M @ x
        for t in range(1, m + 1):
            act = S >= t
            tp = int(y[act].sum()); fpc = int((1 - y[act]).sum())
            U = r_tp*tp + r_fn*(P-tp) + r_fp*fpc + r_tn*(N-fpc)
            if U > best_U: best_U = U
            if t == 1 and U > best_t1: best_t1 = U
            if fpc == 0 and U > best_fp0: best_fp0 = U
    # minimum cardinality attaining best_U
    cmin = m + 1
    for bits in itertools.product([0, 1], repeat=m):
        x = np.array(bits)
        if x.sum() >= cmin: continue
        S = M @ x
        for t in range(1, m + 1):
            act = S >= t
            tp = int(y[act].sum()); fpc = int((1 - y[act]).sum())
            U = r_tp*tp + r_fn*(P-tp) + r_fp*fpc + r_tn*(N-fpc)
            if U == best_U: cmin = min(cmin, int(x.sum())); break
    if best_U == U0 and cmin > m: cmin = 0
    return best_U, best_t1, cmin, best_fp0

fails, total = 0, 0
t0 = time.time()
for inst in range(200):
    m = int(rng.integers(2, 9)); n = int(rng.integers(4, 21))
    dens = rng.uniform(0.15, 0.85)
    y = (rng.random(n) < rng.uniform(0.2, 0.6)).astype(int)
    if y.sum() == 0 or y.sum() == n:
        y[0] = 1 - y[0]
    M = (rng.random((n, m)) < dens).astype(np.int8)
    lam_m, lam_p = rng.choice([2, 4, 8]), rng.choice([2, 4, 8])
    P, N = int(y.sum()), int(n - y.sum())
    costs = costs_int(P, N, lam_m, lam_p)
    eU, eT1, eCmin, eFp0 = enum_all(M, y, costs)
    lp = LabelProblem(M, y, costs)
    s = lp.solve(); s1 = lp.solve_t1()
    sc = lp.solve_min_card(s["U"]); sf = lp.solve_fp0()
    ok = (s["U"] == eU and s1["U"] == eT1
          and sc["Jmin"] == eCmin and sf["U"] == eFp0)
    total += 1
    if not ok:
        fails += 1
        print(f"FAILURE inst={inst} m={m} n={n} lam=({lam_m},{lam_p}) "
              f"milp=({s['U']},{s1['U']},{sc['Jmin']},{sf['U']}) "
              f"enum=({eU},{eT1},{eCmin},{eFp0})")
print(f"QA: {total-fails}/{total} instances EXACT in {time.time()-t0:.0f}s"
      + ("  <<< FAILURES PRESENT" if fails else "  (all agree)"))
