"""Exhaustive enumeration over profiles: baseline against the ILP.

For each cell (label x regime x readout) of medical, enumerates ALL the
decisions of the family: 2^m subsets (including the empty one = null
decision) x integer thresholds t in {1..m}, over the SAME profile
aggregation and the SAME integer scale as the campaign (audit_core2).
Reports the enumeration optimum, the wall time per cell, and the identity
with the archived ILP battery (every medical cell has a proven optimum).

Not a measurement of the paper until it runs on the declared workstation.

Usage: enum_profiles.py --register REG.npz --out OUT.json
       [--battery BATTERY.json] [--labels a,b,c] [--chunk 8192]
"""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit_core2 import costs_int, to_norm, build_M_tau, aggregate, REGIMES


def unpad(fp):
    return [f[f >= 0] for f in fp]


def enum_cell(M, y, costs, chunk=8192):
    """Exact optimum by complete enumeration (internal integer scale)."""
    t0 = time.perf_counter()
    r_tp, r_tn, r_fn, r_fp = (int(c) for c in costs)
    Mp, yp, w = aggregate(M, y)
    P, N = int(y.sum()), int((1 - y).sum())
    U0 = r_tn * N + r_fn * P
    gain = np.where(yp == 1, r_tp - r_fn, r_fp - r_tn).astype(np.float64) * w
    Pn, m = Mp.shape
    best_du, best_mask, best_t = 0.0, 0, 0          # empty subset
    Mp32 = Mp.astype(np.int32)
    total = 1 << m
    for lo in range(1, total, chunk):
        hi = min(lo + chunk, total)
        masks = np.arange(lo, hi, dtype=np.int64)
        X = ((masks[None, :] >> np.arange(m)[:, None]) & 1).astype(np.int32)
        S = Mp32 @ X                                  # (Pn, B) vote counts
        du = np.full(hi - lo, -np.inf)
        tt = np.zeros(hi - lo, dtype=np.int64)
        for t in range(1, m + 1):
            v = gain @ (S >= t)
            upd = v > du
            du[upd] = v[upd]
            tt[upd] = t
        j = int(np.argmax(du))
        if du[j] > best_du:
            best_du, best_mask, best_t = float(du[j]), int(masks[j]), int(tt[j])
    return {"U_int": int(round(U0 + best_du)), "U0_int": int(U0),
            "t": best_t, "J": bin(best_mask).count("1"),
            "mask": best_mask, "profiles": int(Pn),
            "decisions": int(total) * m,
            "t_enum": time.perf_counter() - t0}


def one_label(reg_path, n, k, chunk):
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
            rk[f"{rname}_{reading}"] = enum_cell(M, yk, costs, chunk)
    return str(k), rk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--battery", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--chunk", type=int, default=8192)
    a = ap.parse_args()

    reg = np.load(a.register, allow_pickle=True)
    n, L = reg["Y"].shape
    ks = ([int(x) for x in a.labels.split(",")] if a.labels
          else list(range(L)))
    # eligibility identical to the campaign: >=10 positives and >=10 negatives
    ks = [k for k in ks
          if 10 <= int(reg["Y"][:, k].sum()) <= n - 10]
    t_all = time.perf_counter()
    cells = {}
    for k in ks:
        kk, rk = one_label(a.register, n, k, a.chunk)
        cells[kk] = rk
        print(f"label {kk}: "
              + " ".join(f"{c}={rk[c]['t_enum']:.1f}s"
                         for c in rk if isinstance(rk[c], dict)), flush=True)

    out = {"meta": {"register": Path(a.register).name,
                    "what": "enumeracao exaustiva por perfis (item 48)",
                    "chunk": a.chunk,
                    "total_seconds": time.perf_counter() - t_all},
           "labels": cells}

    if a.battery:
        bat = json.load(open(a.battery))["labels"]
        keys = [f"{r}_{rd}" for r in REGIMES for rd in ("m05", "mtau")]
        agree, diff, times = 0, [], []
        for k, rk in cells.items():
            for key in keys:
                e, p = rk.get(key), bat.get(k, {}).get(key)
                if not (isinstance(e, dict) and isinstance(p, dict)):
                    continue
                if str(p.get("status", "optimal")).startswith("optimal"):
                    if e["U_int"] == p["U_int"]:
                        agree += 1
                    else:
                        diff.append({"k": k, "cell": key,
                                     "enum": e["U_int"], "pli": p["U_int"]})
                times.append((e["t_enum"], p.get("t_solver")))
        out["vs_pli"] = {"identicos": agree, "divergencias": diff,
                         "enum_seconds_median": float(np.median(
                             [t[0] for t in times])),
                         "enum_seconds_max": float(max(
                             t[0] for t in times))}
        if all(t[1] is not None for t in times):
            out["vs_pli"]["pli_seconds_median"] = float(np.median(
                [t[1] for t in times]))
            out["vs_pli"]["pli_seconds_max"] = float(max(
                t[1] for t in times))
        print("VS_PLI:", json.dumps(out["vs_pli"])[:400], flush=True)
        assert not diff, f"DIVERGENCE enum vs ILP: {diff[:3]}"

    json.dump(out, open(a.out, "w"), indent=1)
    print(f"DONE {out['meta']['total_seconds']:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
