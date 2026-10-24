"""
OPERA — compact MIP formulation (paper version).

Implements exactly the model of Section 4:

    variables
        x[k][j] in {0,1}   classifier j selected for label k
        t[k]    integer    consensus threshold, 1 <= t[k] <= tbar[k]
        z[k][i] in {0,1}   instance i activated on label k  (S_ik >= t_k)

    objective
        max  U0 + sum_k [ sum_{i in I+_k} (r_tp - r_fn) z
                        + sum_{i in I-_k} (r_fp - r_tn) z ]

    activation constraints (two per instance-label pair)
        S_ik >= t_k - tbar_k (1 - z_ik)
        S_ik <= t_k - 1 + q_ik z_ik

    structural constants
        q_ik   = sum_j M[i][j][k]          maximum attainable votes
        qbar_k = max_i q_ik
        tbar_k = max(1, qbar_k)

Two backends, both exact:
  * "highs"  — HiGHS via scipy.optimize.milp (the paper's solver of record)
  * "cplex"  — DOcplex with indicator constraints (the paper's cross-check)

Both solve the same integer program and must return the same optimum.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

Backend = Literal["cplex", "highs"]


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------
@dataclass
class Instance:
    """Data of one audit instance.

    M      : binary array (n, m, c) — M[i, j, k] = prediction of member j on record i, label k
    y      : binary array (n, c) — ground-truth labels
    costs  : array (c, 4) — per label, [r_tp, r_tn, r_fn, r_fp]
             convention: r_tp >= 0, r_tn >= 0, r_fn <= 0, r_fp <= 0
    """

    M: np.ndarray
    y: np.ndarray
    costs: np.ndarray
    name: str = "instance"

    def __post_init__(self) -> None:
        self.M = np.asarray(self.M, dtype=np.int8)
        self.y = np.asarray(self.y, dtype=np.int8)
        self.costs = np.asarray(self.costs, dtype=float)
        if self.M.ndim == 2:  # (n, m) -> single label
            self.M = self.M[:, :, None]
        if self.y.ndim == 1:
            self.y = self.y[:, None]
        if self.costs.ndim == 1:
            self.costs = self.costs[None, :]
        n, m, c = self.M.shape
        assert self.y.shape == (n, c), f"y {self.y.shape} incompatible with M {self.M.shape}"
        assert self.costs.shape == (c, 4), f"costs deve ser ({c}, 4)"

    @property
    def n(self) -> int:
        return self.M.shape[0]

    @property
    def m(self) -> int:
        return self.M.shape[1]

    @property
    def c(self) -> int:
        return self.M.shape[2]

    # --- structural constants (constant-hierarchy remark of the paper) ---
    def q(self, k: int) -> np.ndarray:
        """q_ik = maximum number of votes record i can receive on label k."""
        return self.M[:, :, k].sum(axis=1).astype(int)

    def tbar(self, k: int) -> int:
        """tbar_k = max(1, qbar_k), reduced threshold domain."""
        return int(max(1, self.q(k).max()))

    def U0(self) -> float:
        """Utility of the never-flag policy: U0 = sum_k (r_tn |I-_k| + r_fn |I+_k|)."""
        total = 0.0
        for k in range(self.c):
            pos = int(self.y[:, k].sum())
            neg = self.n - pos
            _, r_tn, r_fn, _ = self.costs[k]
            total += r_tn * neg + r_fn * pos
        return float(total)

    def dominance_threshold(self, k: int = 0) -> float:
        """Threshold on |r_fn| above which a single false negative makes U negative.

        Condition:  |r_fn| > r_tp |I+| + r_tn |I-|
        """
        pos = int(self.y[:, k].sum())
        neg = self.n - pos
        r_tp, r_tn, _, _ = self.costs[k]
        return float(r_tp * pos + r_tn * neg)

    def utility_of(self, sel: Sequence[int], t: int, k: int = 0) -> float:
        """Utility of an explicit decision (subset, threshold) on one label."""
        r_tp, r_tn, r_fn, r_fp = self.costs[k]
        if len(sel) == 0:
            S = np.zeros(self.n, dtype=int)
        else:
            S = self.M[:, list(sel), k].sum(axis=1)
        pred = (S >= t).astype(int)
        yk = self.y[:, k]
        tp = int(((pred == 1) & (yk == 1)).sum())
        fn = int(((pred == 0) & (yk == 1)).sum())
        tn = int(((pred == 0) & (yk == 0)).sum())
        fp = int(((pred == 1) & (yk == 0)).sum())
        return float(r_tp * tp + r_tn * tn + r_fn * fn + r_fp * fp)


@dataclass
class Solution:
    U_star: float
    U0: float
    delta: float
    selected: list[list[int]]
    thresholds: list[int]
    confusion: list[dict]
    seconds: float
    backend: str
    status: str
    gap: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        return "go" if self.U_star > 0 else "no-go"

    @property
    def pool_contributes(self) -> bool:
        return self.delta > 0


# ---------------------------------------------------------------------------
# HiGHS backend (scipy.optimize.milp) — exact, no commercial dependency
# ---------------------------------------------------------------------------
def _solve_highs(inst: Instance, time_limit: float | None, verbose: bool) -> Solution:
    from scipy.optimize import LinearConstraint, milp, Bounds
    from scipy.sparse import lil_matrix

    t0 = time.perf_counter()
    n, m, c = inst.n, inst.m, inst.c

    # layout: [x (m*c)] [t (c)] [z (n*c)]
    nx, nt, nz = m * c, c, n * c
    N = nx + nt + nz
    ix = lambda k, j: k * m + j
    it = lambda k: nx + k
    iz = lambda k, i: nx + nt + k * n + i

    obj = np.zeros(N)
    const = inst.U0()
    for k in range(c):
        r_tp, r_tn, r_fn, r_fp = inst.costs[k]
        yk = inst.y[:, k]
        for i in range(n):
            obj[iz(k, i)] = (r_tp - r_fn) if yk[i] == 1 else (r_fp - r_tn)

    rows, lb_c, ub_c = [], [], []
    A = lil_matrix((2 * n * c, N))
    r = 0
    for k in range(c):
        q = inst.q(k)
        tb = inst.tbar(k)
        for i in range(n):
            # S_ik >= t_k - tbar_k (1 - z_ik)   <=>   S - t - tbar*z >= -tbar
            #   z=1 -> S >= t          (ativa)
            #   z=0 -> S >= t - tbar   (inativa)
            for j in range(m):
                if inst.M[i, j, k]:
                    A[r, ix(k, j)] = 1.0
            A[r, it(k)] = -1.0
            A[r, iz(k, i)] = -float(tb)
            lb_c.append(-float(tb)); ub_c.append(np.inf); r += 1
            # S_ik <= t_k - 1 + q_ik z_ik  <=>  S - t - q*z <= -1
            for j in range(m):
                if inst.M[i, j, k]:
                    A[r, ix(k, j)] = 1.0
            A[r, it(k)] = -1.0
            A[r, iz(k, i)] = -float(q[i])
            lb_c.append(-np.inf); ub_c.append(-1.0); r += 1

    lb = np.zeros(N); ub = np.ones(N)
    for k in range(c):
        lb[it(k)] = 1.0
        ub[it(k)] = float(inst.tbar(k))

    integrality = np.ones(N)  # all integer (x, z binary via bounds; t integer)
    # mip_rel_gap = 0: with integer costs the objective is integer, so any
    # absolute gap below 1 already proves optimality; zeroing the relative
    # gap guarantees both backends certify the same optimal value.
    opts = {"disp": verbose, "mip_rel_gap": 0.0}
    if time_limit:
        opts["time_limit"] = time_limit

    res = milp(
        c=-obj,  # milp minimiza
        constraints=LinearConstraint(A.tocsr(), lb_c, ub_c),
        integrality=integrality,
        bounds=Bounds(lb, ub),
        options=opts,
    )
    secs = time.perf_counter() - t0
    if res.x is None:
        raise RuntimeError(f"HiGHS returned no solution: {res.message}")

    xv = np.round(res.x).astype(int)
    selected = [[j for j in range(m) if xv[ix(k, j)] == 1] for k in range(c)]
    thresholds = [int(xv[it(k)]) for k in range(c)]
    U_star = const + float(obj @ res.x)

    confusion = []
    for k in range(c):
        S = (inst.M[:, selected[k], k].sum(axis=1) if selected[k]
             else np.zeros(n, dtype=int))
        pred = (S >= thresholds[k]).astype(int)
        yk = inst.y[:, k]
        confusion.append(dict(
            tp=int(((pred == 1) & (yk == 1)).sum()),
            fn=int(((pred == 0) & (yk == 1)).sum()),
            tn=int(((pred == 0) & (yk == 0)).sum()),
            fp=int(((pred == 1) & (yk == 0)).sum()),
        ))

    u0 = inst.U0()
    return Solution(U_star=U_star, U0=u0, delta=U_star - u0, selected=selected,
                    thresholds=thresholds, confusion=confusion, seconds=secs,
                    backend="highs", status=str(res.status),
                    extra={"message": res.message})


# ---------------------------------------------------------------------------
# CPLEX backend (DOcplex, indicator constraints) — the paper cross-check
# ---------------------------------------------------------------------------
def _solve_cplex(inst: Instance, time_limit: float | None, verbose: bool) -> Solution:
    from docplex.mp.model import Model

    t0 = time.perf_counter()
    n, m, c = inst.n, inst.m, inst.c
    with Model(name="opera", log_output=verbose) as mdl:
        x = {(k, j): mdl.binary_var(name=f"x_{k}_{j}") for k in range(c) for j in range(m)}
        t = {k: mdl.integer_var(lb=1, ub=inst.tbar(k), name=f"t_{k}") for k in range(c)}
        z = {(k, i): mdl.binary_var(name=f"z_{k}_{i}") for k in range(c) for i in range(n)}

        for k in range(c):
            for i in range(n):
                S = mdl.sum(x[k, j] for j in range(m) if inst.M[i, j, k])
                # indicator constraints (formulation reported in the paper)
                mdl.add_indicator(z[k, i], S - t[k] >= 0, active_value=1)
                mdl.add_indicator(z[k, i], t[k] - S >= 1, active_value=0)

        terms = []
        for k in range(c):
            r_tp, r_tn, r_fn, r_fp = inst.costs[k]
            yk = inst.y[:, k]
            for i in range(n):
                coef = (r_tp - r_fn) if yk[i] == 1 else (r_fp - r_tn)
                terms.append(coef * z[k, i])
        mdl.maximize(inst.U0() + mdl.sum(terms))

        # Zero optimality tolerances. With integer costs the objective is
        # integer, so any absolute gap below 1 already proves optimality;
        # zeroing removes even the theoretical possibility of an early stop
        # under CPLEX's default relative gap of 1e-4, guaranteeing that both
        # backends certify the same optimal objective value.
        mdl.parameters.mip.tolerances.mipgap = 0.0
        mdl.parameters.mip.tolerances.absmipgap = 0.0
        if time_limit:
            mdl.parameters.timelimit = time_limit
        sol = mdl.solve()
        secs = time.perf_counter() - t0
        if sol is None:
            raise RuntimeError("CPLEX found no solution")

        selected = [[j for j in range(m) if sol.get_value(x[k, j]) > 0.5] for k in range(c)]
        thresholds = [int(round(sol.get_value(t[k]))) for k in range(c)]
        U_star = float(sol.get_objective_value())

    confusion = []
    for k in range(c):
        S = (inst.M[:, selected[k], k].sum(axis=1) if selected[k]
             else np.zeros(n, dtype=int))
        pred = (S >= thresholds[k]).astype(int)
        yk = inst.y[:, k]
        confusion.append(dict(
            tp=int(((pred == 1) & (yk == 1)).sum()),
            fn=int(((pred == 0) & (yk == 1)).sum()),
            tn=int(((pred == 0) & (yk == 0)).sum()),
            fp=int(((pred == 1) & (yk == 0)).sum()),
        ))
    u0 = inst.U0()
    return Solution(U_star=U_star, U0=u0, delta=U_star - u0, selected=selected,
                    thresholds=thresholds, confusion=confusion, seconds=secs,
                    backend="cplex", status="optimal")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def solve(inst: Instance, backend: Backend = "highs",
          time_limit: float | None = None, verbose: bool = False) -> Solution:
    """Solve the OPERA audit. Exact backends: 'highs' or 'cplex'."""
    if backend == "cplex":
        return _solve_cplex(inst, time_limit, verbose)
    if backend == "highs":
        return _solve_highs(inst, time_limit, verbose)
    raise ValueError(f"backend desconhecido: {backend}")


def brute_force(inst: Instance, k: int = 0) -> tuple[float, list[int], int]:
    """Exhaustive enumeration (small m only). Used to test the MIP."""
    from itertools import combinations

    best = (-np.inf, [], 1)
    for r in range(inst.m + 1):
        for sel in combinations(range(inst.m), r):
            for t in range(1, inst.tbar(k) + 1):
                u = inst.utility_of(sel, t, k)
                if u > best[0]:
                    best = (u, list(sel), t)
    return best
