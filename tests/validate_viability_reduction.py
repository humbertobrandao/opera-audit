"""Exhaustive validation of the viability-hardness construction
(NP-completeness of deciding U* > 0).

For each graph G=(V,E) without isolated vertices and each target q in
1..N: builds the OPERA instance of the reduction (2N classifiers a_j,
b_j; instances: positive p_j with votes {a_j, b_j}; edge negative e_uv
with votes {a_u, a_v}; auxiliary negative f_j with vote {b_j}; payoffs
r_tp = 2N-2q+1, r_tn = 0, r_fn = -(2q-1), r_fp = -2N(N+1)) and checks,
by enumeration of ALL selections S and thresholds t:

  (1) U* = 2N*alpha(G) - N(2q-1)            [value identity]
  (2) U* > 0  <=>  alpha(G) >= q            [sign equivalence]
  (3) the same U* and the same equivalence with t FIXED at 2
  (4) OPERA's sign hypotheses: r_tp>0, r_tn>=0, r_fn<0, r_fp<0.

Graphs: all labeled graphs without isolated vertices with n = 2, 3, 4
(exhaustive) and random samples with n = 5 and 6. alpha(G) by vertex
enumeration.
"""
import itertools, sys
import numpy as np

rng = np.random.default_rng(20261015)


def alpha(n, edges):
    best = 0
    for mask in range(1 << n):
        S = [v for v in range(n) if mask >> v & 1]
        if all(not (u in S and v in S) for u, v in edges):
            best = max(best, len(S))
    return best


def build(n, edges, q):
    """Matrix M (instances x 2n classifiers), y, payoffs."""
    m = 2 * n                       # columns: a_0..a_{n-1}, b_0..b_{n-1}
    rows, y = [], []
    for j in range(n):              # positives p_j: votes a_j and b_j
        r = np.zeros(m, dtype=np.int8); r[j] = 1; r[n + j] = 1
        rows.append(r); y.append(1)
    for (u, v) in edges:            # edge negatives: a_u and a_v
        r = np.zeros(m, dtype=np.int8); r[u] = 1; r[v] = 1
        rows.append(r); y.append(0)
    for j in range(n):              # auxiliary negatives f_j: b_j
        r = np.zeros(m, dtype=np.int8); r[n + j] = 1
        rows.append(r); y.append(0)
    M = np.array(rows); y = np.array(y)
    r_tp, r_tn = 2 * n - 2 * q + 1, 0
    r_fn, r_fp = -(2 * q - 1), -2 * n * (n + 1)
    assert r_tp > 0 and r_tn >= 0 and r_fn < 0 and r_fp < 0   # (4)
    return M, y, (r_tp, r_tn, r_fn, r_fp)


def oracle(M, y, costs, t_fix=None):
    """U* by enumeration of all (S, t)."""
    r_tp, r_tn, r_fn, r_fp = costs
    nI, m = M.shape
    base_pos = np.where(y == 1, r_fn, r_tn).astype(np.int64)   # nothing activated
    gain = np.where(y == 1, r_tp - r_fn, r_fp - r_tn).astype(np.int64)
    best = base_pos.sum() if t_fix is None else None   # empty S, any t
    if t_fix is not None:
        best = base_pos.sum()        # empty S with t=2: nothing activates
    for mask in range(1, 1 << m):
        sel = [j for j in range(m) if mask >> j & 1]
        sc = M[:, sel].sum(axis=1)
        ts = (t_fix,) if t_fix is not None else range(1, len(sel) + 1)
        for t in ts:
            act = sc >= t
            U = base_pos.sum() + gain[act].sum()
            if U > best:
                best = U
    return int(best)


def all_graphs_no_isolated(n):
    pairs = list(itertools.combinations(range(n), 2))
    for emask in range(1, 1 << len(pairs)):
        edges = [pairs[i] for i in range(len(pairs)) if emask >> i & 1]
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1; deg[v] += 1
        if all(d > 0 for d in deg):
            yield edges


def sample_graphs(n, k):
    pairs = list(itertools.combinations(range(n), 2))
    out, seen = [], set()
    while len(out) < k:
        emask = int(rng.integers(1, 1 << len(pairs)))
        if emask in seen:
            continue
        seen.add(emask)
        edges = [pairs[i] for i in range(len(pairs)) if emask >> i & 1]
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1; deg[v] += 1
        if all(d > 0 for d in deg):
            out.append(edges)
    return out


def check(n, edges):
    a = alpha(n, edges)
    for q in range(1, n + 1):
        M, y, costs = build(n, edges, q)
        Ustar = oracle(M, y, costs)
        target = 2 * n * a - n * (2 * q - 1)
        ok1 = (Ustar == target)                      # (1)
        ok2 = ((Ustar > 0) == (a >= q))              # (2)
        U2 = oracle(M, y, costs, t_fix=2)
        ok3 = (U2 == target) and ((U2 > 0) == (a >= q))  # (3)
        if not (ok1 and ok2 and ok3):
            print(f"FAILURE n={n} E={edges} q={q}: U*={Ustar} target={target} "
                  f"alpha={a} U2={U2}")
            return False
    return True


tot = bad = 0
for n in (2, 3, 4):
    for edges in all_graphs_no_isolated(n):
        tot += 1
        if not check(n, edges):
            bad += 1
    print(f"n={n}: exhaustive ok ({tot} graphs accumulated)")
for n, k in ((5, 200), (6, 60)):
    for edges in sample_graphs(n, k):
        tot += 1
        if not check(n, edges):
            bad += 1
    print(f"n={n}: sample of {k} ok")
print(f"{'VALIDATED' if bad == 0 else 'FAILED'}: {tot} graphs x all "
      f"targets q; {bad} violations of identities (1)-(3); sign "
      f"hypotheses (4) checked on every instance.")
sys.exit(1 if bad else 0)
