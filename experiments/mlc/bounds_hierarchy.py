"""Polynomial hierarchy of bounds (B2): where the difficulty is born.

For each cell (label x regime x readout), over the SAME profile
aggregation and integer scale as the campaign (audit_core2), computes
three polynomial upper bounds on U*:

  U_B    best Boolean rule over the vote profiles (any deterministic
         combiner of the binary predictions, including stacking and
         weighted voting over the votes):
         U_B = U0 + sum_v (w_v)+, with w_v = a*P_v - b*N_v,
         a = r_tp - r_fn >= 0, b = r_tn - r_fp >= 0. Closed form.
  U_B0   same, restricted to the rules that do not flag the null vote
         profile (every rule of OPERA's family satisfies this, as t >= 1).
  U_mon  best MONOTONE rule over the nonnull profiles (includes weighted
         voting with weights >= 0): maximum-weight upper set of profiles
         = maximal closure, solved by a minimum cut (Picard, 1976).
         Pure-Python Dinic (profiles ~10^2-10^3).

Guaranteed chain: U* <= U_mon <= U_B0 <= U_B <= reachability bound.
The side useful as a CERTIFICATE is the no-go: U_B <= 0 excludes any
deterministic combiner; U_mon <= 0 excludes the monotone rules.
With --battery, checks the chain against the ILP's U* (proven cells).

Usage: hierarquia_cotas.py --register REG.npz --out OUT.json
       [--battery BAT.json] [--labels a,b,c]
"""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit_core2 import costs_int, to_norm, build_M_tau, aggregate, REGIMES


def unpad(fp):
    return [f[f >= 0] for f in fp]


class Dinic:
    def __init__(self, n):
        self.n = n; self.g = [[] for _ in range(n)]
    def add(self, u, v, cap):
        self.g[u].append([v, cap, len(self.g[v])])
        self.g[v].append([u, 0, len(self.g[u]) - 1])
    def maxflow(self, s, t):
        flow = 0
        while True:
            level = [-1] * self.n; level[s] = 0; q = [s]
            for u in q:
                for e in self.g[u]:
                    if e[1] > 0 and level[e[0]] < 0:
                        level[e[0]] = level[u] + 1; q.append(e[0])
            if level[t] < 0:
                return flow
            it = [0] * self.n
            def dfs(u, f):
                if u == t: return f
                while it[u] < len(self.g[u]):
                    e = self.g[u][it[u]]
                    if e[1] > 0 and level[e[0]] == level[u] + 1:
                        d = dfs(e[0], min(f, e[1]))
                        if d > 0:
                            e[1] -= d; self.g[e[0]][e[2]][1] += d
                            return d
                    it[u] += 1
                return 0
            while True:
                f = dfs(s, float("inf"))
                if f == 0: break
                flow += f


def max_upset(Mp, w):
    """Maximum of sum w_v over upper sets (up-sets) of the profiles
    (componentwise dominance). Maximal closure via minimum cut: if v is
    in the set and v' >= v, then v' is in the set too."""
    P = len(w)
    pos = sum(x for x in w if x > 0)
    if pos == 0:
        return 0
    s, t = P, P + 1
    din = Dinic(P + 2)
    INF = float("inf")
    for i in range(P):
        if w[i] > 0: din.add(s, i, w[i])
        elif w[i] < 0: din.add(i, t, -w[i])
    dom = Mp.astype(bool)
    for i in range(P):
        for j in range(P):
            if i != j and np.all(dom[j] >= dom[i]) and np.any(dom[j] != dom[i]):
                din.add(i, j, INF)   # i no up-set => j no up-set
    return pos - din.maxflow(s, t)


def bounds_cell(M, y, costs):
    t0 = time.perf_counter()
    r_tp, r_tn, r_fn, r_fp = (int(c) for c in costs)
    a, b = r_tp - r_fn, r_tn - r_fp          # >= 0 pela convencao de sinais
    Mp, yp, wgt = aggregate(M, y)
    P, N = int(y.sum()), int((1 - y).sum())
    U0 = r_tn * N + r_fn * P
    # aggregates by VOTE profile (ignoring y): w_v = a*P_v - b*N_v
    key = {}
    for i in range(Mp.shape[0]):
        k = Mp[i].tobytes()
        if k not in key:
            key[k] = [Mp[i], 0]
        key[k][1] += int(wgt[i]) * (a if yp[i] == 1 else -b)
    profs = list(key.values())
    Vp = np.array([p[0] for p in profs], dtype=np.int8)
    w = [p[1] for p in profs]
    nz = [i for i in range(len(w)) if Vp[i].any()]
    U_B  = U0 + sum(x for x in w if x > 0)
    U_B0 = U0 + sum(w[i] for i in nz if w[i] > 0)
    U_mon = U0 + max_upset(Vp[nz], [w[i] for i in nz])
    return {"U_B_int": int(U_B), "U_B0_int": int(U_B0),
            "U_mon_int": int(U_mon), "U0_int": int(U0),
            "profiles_voto": len(profs),
            "nogo_B": U_B <= 0, "nogo_B0": U_B0 <= 0,
            "nogo_mon": U_mon <= 0,
            "t_bounds": time.perf_counter() - t0}


def one_label(reg_path, n, k):
    reg = np.load(reg_path, allow_pickle=True)
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
            rk[f"{rname}_{reading}"] = bounds_cell(M, yk, costs)
    return str(k), rk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--battery", default=None)
    ap.add_argument("--labels", default=None)
    a = ap.parse_args()
    reg = np.load(a.register, allow_pickle=True)
    n, L = reg["Y"].shape
    ks = ([int(x) for x in a.labels.split(",")] if a.labels else range(L))
    ks = [k for k in ks if 10 <= int(reg["Y"][:, k].sum()) <= n - 10]
    t_all = time.perf_counter()
    cells = {}
    for k in ks:
        kk, rk = one_label(a.register, n, k)
        cells[kk] = rk
    out = {"meta": {"register": Path(a.register).name,
                    "what": "B2 bounds hierarchy (U_B, U_B0, U_mon)",
                    "total_seconds": time.perf_counter() - t_all},
           "labels": cells}
    if a.battery:
        bat = json.load(open(a.battery))["labels"]
        keys = [f"{r}_{rd}" for r in REGIMES for rd in ("m05", "mtau")]
        chain_viol, nogo = [], {"B": 0, "B0": 0, "mon": 0,
                                "pli_nogo_cells": 0, "cells": 0}
        for k, rk in cells.items():
            for key in keys:
                e, p = rk.get(key), bat.get(k, {}).get(key)
                if not (isinstance(e, dict) and isinstance(p, dict)):
                    continue
                nogo["cells"] += 1
                if p["verdict"] == "no-go":
                    nogo["pli_nogo_cells"] += 1
                for nm in ("B", "B0", "mon"):
                    nogo[nm] += int(e[f"nogo_{nm}"])
                opt = str(p.get("status", "optimal")).startswith("optimal")
                U = p["U_int"]
                ok = (e["U_mon_int"] <= e["U_B0_int"] <= e["U_B_int"]
                      and (not opt or U <= e["U_mon_int"]))
                if not ok:
                    chain_viol.append({"k": k, "cell": key, "U": U,
                                       "mon": e["U_mon_int"],
                                       "B0": e["U_B0_int"],
                                       "B": e["U_B_int"]})
        out["vs_pli"] = {"cadeia_violacoes": chain_viol, "nogo": nogo}
        print("VS_PLI:", json.dumps(out["vs_pli"])[:500], flush=True)
        assert not chain_viol, f"CADEIA VIOLADA: {chain_viol[:3]}"
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"DONE {out['meta']['total_seconds']:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
