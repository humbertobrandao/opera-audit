"""Core v2: scaled INTEGER objective, exact tie-break, G_cons, sequential
baselines A-OR/A-MAJ/B, FP=0 pre-check and full frontier. Internal scale:
multiply the normalized utility by PN/50 => payoffs
(N, P, -lam_minus*N, -lam_plus*P), all integer for lam in {2,4,8}.
Conversion when reporting: U_norm = 50/(PN) * U_int.
"""
from __future__ import annotations
import time
import numpy as np
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds

REGIMES = {"R1": (2, 2), "R2": (4, 2), "R3": (2, 4)}
R4_PROFILES = [(4, 2), (2, 2), (2, 4)]


def costs_int(P, N, lam_m, lam_p):
    """Integer (r_tp, r_tn, r_fn, r_fp) on the internal PN/50 scale."""
    return np.array([N, P, -lam_m * N, -lam_p * P], dtype=np.int64)


def to_norm(U_int, P, N):
    return 50.0 * U_int / (P * N)


def tau_star(P_tr, N_tr, lam_m, lam_p):
    return (1.0 + lam_p) * P_tr / ((1.0 + lam_m) * N_tr + (1.0 + lam_p) * P_tr)


def build_M_tau(probs, y, folds, lam_m, lam_p):
    M = np.zeros(probs.shape, dtype=np.int8)
    for test_idx in folds:
        tr = np.setdiff1d(np.arange(len(y)), test_idx)
        P_tr, N_tr = int(y[tr].sum()), int((1 - y[tr]).sum())
        tau = tau_star(P_tr, N_tr, lam_m, lam_p)
        M[test_idx] = (probs[test_idx] >= tau).astype(np.int8)
    return M


def aggregate(M, y):
    keys, order, wpos = {}, [], []
    for i in range(M.shape[0]):
        kk = (M[i].tobytes(), int(y[i]))
        if kk not in keys:
            keys[kk] = len(order); order.append((M[i].copy(), int(y[i]))); wpos.append(0)
        wpos[keys[kk]] += 1
    Mp = np.array([o[0] for o in order], dtype=np.int8)
    yp = np.array([o[1] for o in order], dtype=np.int8)
    return Mp, yp, np.array(wpos, dtype=np.int64)


class LabelProblem:
    """Aggregated instance of one (label, readout); solves the variants."""

    def __init__(self, M, y, costs):
        self.n, self.m = M.shape
        self.costs = costs
        r_tp, r_tn, r_fn, r_fp = (int(c) for c in costs)
        self.Mp, self.yp, self.w = aggregate(M, y)
        self.Pn = self.Mp.shape[0]
        self.q = self.Mp.sum(axis=1).astype(float)
        self.Pk, self.Nk = int(y.sum()), int((1 - y).sum())
        self.U0 = r_tn * self.Nk + r_fn * self.Pk
        self.gain = np.where(self.yp == 1, r_tp - r_fn, r_fp - r_tn) * self.w
        self.M, self.y = M, y

    # ---- MILP construction: vars x(m) | t(1) | z(Pn) ----
    def _model(self, t_fix=None, card_fix=None, forbid_neg=False,
               U_floor=None, obj="U", gain=None):
        m, Pn, tbar = self.m, self.Pn, float(self.m)
        g = self.gain if gain is None else gain
        nv = m + 1 + Pn
        A_rows, lb, ub = [], [], []
        for p in range(Pn):
            row = np.zeros(nv)
            row[:m] = self.Mp[p]; row[m] = -1.0; row[m + 1 + p] = -tbar
            A_rows.append(row); lb.append(-tbar); ub.append(np.inf)
            row2 = np.zeros(nv)
            row2[:m] = self.Mp[p]; row2[m] = -1.0; row2[m + 1 + p] = -self.q[p]
            A_rows.append(row2); lb.append(-np.inf); ub.append(-1.0)
        if card_fix is not None:
            row = np.zeros(nv); row[:m] = 1.0
            A_rows.append(row); lb.append(card_fix); ub.append(card_fix)
        if U_floor is not None:
            row = np.zeros(nv); row[m + 1:] = g
            A_rows.append(row); lb.append(U_floor - self.U0); ub.append(np.inf)
        c = np.zeros(nv)
        if obj == "U":
            c[m + 1:] = -g
        elif obj == "card":
            c[:m] = 1.0
        lo = np.r_[np.zeros(m), 1.0, np.zeros(Pn)]
        hi = np.r_[np.ones(m), float(self.m), np.ones(Pn)]
        if t_fix is not None:
            lo[m] = hi[m] = float(t_fix)
        if forbid_neg:
            hi[m + 1:][self.yp == 0] = 0.0
        A = sparse.csr_matrix(np.array(A_rows))
        return c, LinearConstraint(A, lb, ub), Bounds(lo, hi), nv

    def _note_status(self, optimal):
        """Sticky record per instance (label x regime x readout):
        'optimal' only if NO solve of the block hit the TL; otherwise
        'time_limit(n)' with the count of affected solves."""
        if optimal:
            if getattr(self, "_tl_hits", 0) == 0:
                self.last_status = "optimal"
        else:
            self._tl_hits = getattr(self, "_tl_hits", 0) + 1
            self.last_status = f"time_limit({self._tl_hits})"

    def _run(self, relax=False, **kw):
        c, cons, bounds, nv = self._model(**kw)
        t0 = time.perf_counter()
        res = milp(c=c, constraints=cons,
                   integrality=(np.zeros(nv) if relax else np.ones(nv)),
                   bounds=bounds,
                   options={"mip_rel_gap": 0.0, "time_limit": 600.0})
        dt = time.perf_counter() - t0
        if res.status == 0:                    # certified optimum
            self._note_status(True)
        elif res.status == 1 and res.x is not None:
            self._note_status(False)           # TL with an incumbent: record and continue
        else:
            # TL without any HiGHS incumbent: fall back to the trivial feasible
            # solution (signal nothing, U = U0). The cell is recorded as
            # resource-limited ("time_limit(...);trivial") and excluded from the
            # exact-match criterion by compare_results.py; bound coherence is
            # preserved (any optimum >= the trivial U0).
            import types
            self._note_status(False)
            self.last_status += ";trivial"
            xtr = np.zeros(nv); xtr[self.m] = 1.0
            res = types.SimpleNamespace(x=xtr, mip_node_count=0,
                                        status=1, message=res.message)
        return res, dt

    def _extract(self, res, dt, gain=None):
        g = self.gain if gain is None else gain
        m = self.m
        xv = np.round(res.x[:m]).astype(int)
        tv = int(round(res.x[m]))
        zv = np.round(res.x[m + 1:]).astype(int)
        U = float(self.U0 + (g * zv).sum())   # exact integer when g is integer
        tp = int(self.w[(zv == 1) & (self.yp == 1)].sum())
        fp = int(self.w[(zv == 1) & (self.yp == 0)].sum())
        return {"U": U, "U0": self.U0, "dU": U - self.U0,
                "verdict": "go" if U > 0 else "no-go",
                "J": int(xv.sum()), "t": tv, "tp": tp, "fp": fp,
                "fn": self.Pk - tp, "tn": self.Nk - fp, "seconds": dt,
                "nodes": int(getattr(res, "mip_node_count", -1) or 0),
                "profiles": self.Pn, "x": xv}

    # ---- protocol solves ----
    def solve(self):                       # main (joint) audit
        res, dt = self._run()
        # keeps the optimal solution as a feasible warm start for min|J|
        self._warm = np.round(np.asarray(res.x, float))
        return self._extract(res, dt)

    def root_bound(self):
        res, dt = self._run(relax=True)
        return {"root_bound": self.U0 - res.fun, "seconds": dt}

    def solve_t1(self):                    # restricted consensus (OR rule)
        res, dt = self._run(t_fix=1)
        return self._extract(res, dt)

    def solve_min_card(self, U_star):      # EXACT lexicographic tie-break
        # warm start: the main-audit solution is feasible here by
        # construction (it satisfies U >= U*), guaranteeing an incumbent
        # from the start; used by the CPLEX backend, ignored by HiGHS/scipy.
        self._mip_start = getattr(self, "_warm", None)
        try:
            res, dt = self._run(obj="card", U_floor=U_star)
        finally:
            self._mip_start = None
        out = self._extract(res, dt)
        out["Jmin"] = out.pop("J")
        return out

    def solve_fp0(self):                   # pre-check: max U s.t. FP = 0
        res, dt = self._run(forbid_neg=True)
        return self._extract(res, dt)

    def solve_rho(self, rho_num, rho_den=1):  # inflated FP payoff: r_fp*rho
        g = np.where(self.yp == 1,
                     (self.costs[0] - self.costs[2]) * self.w,
                     (self.costs[3] * rho_num / rho_den - self.costs[1]) * self.w)
        res, dt = self._run(gain=g)
        return self._extract(res, dt, gain=g)

    # ---- utility of a fixed committee, enumerated threshold (no MILP) ----
    def enum_t(self, J_mask, costs=None):
        cst = self.costs if costs is None else costs
        r_tp, r_tn, r_fn, r_fp = (int(c) for c in cst)
        S = self.M[:, J_mask.astype(bool)].sum(axis=1)
        best = (self.U0, 0)                # t > max(S): null committee
        for t in range(1, self.m + 1):
            act = S >= t
            tp = int(self.y[act].sum()); fpc = int((1 - self.y[act]).sum())
            U = (r_tp * tp + r_fn * (self.Pk - tp)
                 + r_fp * fpc + r_tn * (self.Nk - fpc))
            if U > best[0]:
                best = (U, t)
        return best                        # (U, t)

    # ---- sequential baselines ----
    def baseline_A_OR(self):
        s1 = self.solve_t1()
        U2, t2 = self.enum_t(s1["x"])
        return {"U": max(s1["U"], U2), "t": t2 if U2 >= s1["U"] else 1,
                "J": s1["J"]}

    def baseline_A_MAJ(self):
        best = None
        for qc in range(1, self.m + 1):
            t_maj = qc // 2 + 1
            res, dt = self._run(t_fix=t_maj, card_fix=qc)
            s = self._extract(res, dt)
            if best is None or s["U"] > best["U"]:
                best = s
        U2, t2 = self.enum_t(best["x"])
        return {"U": max(best["U"], U2), "t": t2 if U2 >= best["U"] else best["t"],
                "J": best["J"]}

    def baseline_B(self):
        U1, t1 = self.enum_t(np.ones(self.m, dtype=bool))
        if t1 == 0:                        # no t helps: null committee
            return {"U": self.U0, "t": 0, "J": 0}
        res, dt = self._run(t_fix=t1)
        s = self._extract(res, dt)
        return {"U": s["U"], "t": t1, "J": s["J"]}

    # ---- full frontier (frontier proposition), fixed M, rho as a fraction ----
    def frontier(self, U_nom, tol_iter=60):
        """rho* such that U*(rho) = 0; cases: no viable region, inf, finite."""
        s0 = self.solve_rho(0)             # U*(0): free FPs
        if s0["U"] <= 0:
            return {"case": "sem_regiao_viavel"}
        sfp0 = self.solve_fp0()
        if sfp0["U"] > 0:
            return {"case": "inf", "U_fp0": sfp0["U"]}
        # Dinkelbach starting at rho = 1
        rho, solves = 1.0, 0
        s = None
        for _ in range(tol_iter):
            s = self.solve_rho(rho) if s is None else s
            b = float(self.costs[3]) * s["fp"]   # coefficient of rho in the active piece
            if b == 0:                     # U>0 without FP: excluded by the pre-check
                return {"case": "inf_degenerado", "rho": rho}
            # active piece: U(r) = s.U + r_fp*fp*(r - rho); root:
            new_rho = rho - s["U"] / b
            if abs(new_rho - rho) < 1e-12:
                break
            rho = new_rho
            s = self.solve_rho(rho)
            solves += 1
            if abs(s["U"]) < 1e-9:
                break
        return {"case": "finito", "rho": rho, "solves": solves}
