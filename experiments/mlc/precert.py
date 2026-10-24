"""Cheap pre-certification: what trivial candidates and a simple bound
settle BEFORE any ILP, cell by cell.

For each eligible label x {R1,R2,R3} x {m05, mtau} (same M, costs and
aggregation as audit_core2, with no solve):

  Candidates (lower bounds, O(m*(Pn+m)) vectorized):
    C-null    empty committee (U0; always negative by construction);
    C-single  each classifier alone with t=1 (m candidates);
    C-full    full pool at every threshold t=1..m;
    C-greedy  prefixes nested by decreasing individual utility,
              each prefix at every threshold t<=q.
  Simple upper bound (no solver):
    UB = max_t [ U0 + sum of the gains of the positives with q_i >= t ],
    t in {0(null),1..m}; valid because activating positive i under (S,t)
    requires score_S(i) >= t and score_S(i) <= q_i, and a negative that
    is not activated contributes zero.

  Pre-test verdict: go_pre if the best candidate has U > 0; nogo_pre if
  UB <= 0; open otherwise. go_pre and nogo_pre are exact certificates
  (no tolerance); open only means that the cheap pre-test does not
  decide, and the ILP is needed.

With --battery, compares against the JSON of the battery already run:
verdict agreement, what the ILP added on the open cells, and the
best-candidate/U* ratio where the optimum was proved.

Usage: precert.py --register REG.npz --out OUT.json
       [--battery BATTERY.json] [--jobs N] [--labels a,b,c]
"""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit_core2 import costs_int, to_norm, build_M_tau, aggregate, REGIMES


def unpad(fp):
    return [f[f >= 0] for f in fp]


def best_over_thresholds(scores, gain, m_max):
    """max_t U and argmax for a fixed committee: U(t) = U0 + sum of the
    gains of the profiles with score >= t. Bincount suffix-sum: O(Pn + m_max)."""
    if m_max == 0:
        return 0.0, 0
    sc = np.minimum(scores, m_max).astype(np.int64)
    bc = np.bincount(sc, weights=gain, minlength=m_max + 1)
    sfx = np.cumsum(bc[::-1])[::-1]          # sfx[t] = soma dos ganhos score>=t
    t_best = int(np.argmax(sfx[1:]) + 1)
    return float(sfx[t_best]), t_best        # (U - U0, t)


def precert_cell(M, y, costs):
    """Full pre-test of one cell. Returns a dict with the candidates, the
    bound and the pre-test verdict (all on the internal integer scale)."""
    t0 = time.perf_counter()
    r_tp, r_tn, r_fn, r_fp = (int(c) for c in costs)
    Mp, yp, w = aggregate(M, y)
    Pn, m = Mp.shape
    Pk, Nk = int(y.sum()), int((1 - y).sum())
    U0 = r_tn * Nk + r_fn * Pk
    gain = np.where(yp == 1, r_tp - r_fn, r_fp - r_tn).astype(np.float64) * w
    q = Mp.sum(axis=1).astype(np.int64)      # votes of the full pool

    best = {"name": "C-null", "U_int": int(U0), "t": 0, "J": 0}

    def consider(dU, t, name, J):
        nonlocal best
        Uc = U0 + dU
        if Uc > best["U_int"]:
            best = {"name": name, "U_int": int(round(Uc)), "t": int(t),
                    "J": int(J)}

    # C-single: individual utility of each classifier (t=1)
    # dU_j = sum of the gains of the profiles with Mp[:,j] == 1
    dU_single = (gain[:, None] * Mp).sum(axis=0)       # (m,)
    j_best = int(np.argmax(dU_single))
    consider(float(dU_single[j_best]), 1, "C-single", 1)

    # C-full: full pool, every threshold
    dU_full, t_full = best_over_thresholds(q, gain, m)
    consider(dU_full, t_full, "C-full", m)

    # C-greedy: prefixes by decreasing individual utility
    order = np.argsort(-dU_single, kind="stable")
    cum = np.cumsum(Mp[:, order].astype(np.int64), axis=1)
    for qc in range(1, m + 1):
        dU_q, t_q = best_over_thresholds(cum[:, qc - 1], gain, qc)
        consider(dU_q, t_q, "C-greedy", qc)

    # Simple upper bound: reachable positives per threshold
    gpos = np.where(yp == 1, gain, 0.0)
    sc = np.minimum(q, m).astype(np.int64)
    bc = np.bincount(sc, weights=gpos, minlength=m + 1)
    sfx = np.cumsum(bc[::-1])[::-1]
    UB = U0 + float(np.max(sfx[1:])) if m >= 1 else float(U0)
    UB = max(UB, float(U0))                  # the "null" t is also a case

    go_pre = best["U_int"] > 0
    nogo_pre = (not go_pre) and (UB <= 0)
    return {"U0_int": int(U0),
            "best_cand": best,
            "best_cand_norm": to_norm(best["U_int"], Pk, Nk),
            "ub_simple_int": int(round(UB)),
            "pre_verdict": ("go_pre" if go_pre
                            else "nogo_pre" if nogo_pre else "open"),
            "profiles": int(Pn),
            "t_precert": time.perf_counter() - t0}


def one_label(reg_path, n, k):
    reg = np.load(reg_path, allow_pickle=True)   # lazy read
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
            rk[f"{rname}_{reading}"] = precert_cell(M, yk, costs)
    return str(k), rk


def compare_with_battery(cells, battery_path):
    """Confronts the pre-test with the ILP battery already run."""
    bat = json.load(open(battery_path))
    labels = bat["labels"]
    keys = [f"{r}_{rd}" for r in REGIMES for rd in ("m05", "mtau")]
    summary = {"cells_total": 0, "go_pre": 0, "nogo_pre": 0, "open": 0,
               "open_pli_go": 0, "open_pli_nogo": 0,
               "disagreements": [], "ratios_best_over_Ustar": [],
               "pli_status_tl_in_open": 0}
    for k, rk in cells.items():
        if k not in labels:
            continue
        for key in keys:
            pc, pli = rk.get(key), labels[k].get(key)
            if pc is None or pli is None:
                continue
            summary["cells_total"] += 1
            summary[pc["pre_verdict"]] += 1
            pli_go = pli["verdict"] == "go"
            pli_opt = str(pli.get("status", "optimal")).startswith("optimal")
            # consistency (exact certificates can never contradict):
            if pc["pre_verdict"] == "go_pre" and not pli_go:
                summary["disagreements"].append(
                    {"k": k, "cell": key, "type": "go_pre_vs_pli_nogo"})
            if pc["pre_verdict"] == "nogo_pre" and pli_go:
                summary["disagreements"].append(
                    {"k": k, "cell": key, "type": "nogo_pre_vs_pli_go"})
            if pc["best_cand"]["U_int"] > pli["U_int"] + 0.5 and pli_opt:
                summary["disagreements"].append(
                    {"k": k, "cell": key, "type": "candidate_above_optimum"})
            if pc["pre_verdict"] == "open":
                summary["open_pli_go" if pli_go else "open_pli_nogo"] += 1
                if not pli_opt:
                    summary["pli_status_tl_in_open"] += 1
            if pli_opt and pli["U_int"] > 0:
                summary["ratios_best_over_Ustar"].append(
                    max(0.0, pc["best_cand"]["U_int"]) / pli["U_int"])
            pc["pli"] = {"U_int": pli["U_int"], "verdict": pli["verdict"],
                         "status": pli.get("status", "optimal")}
    rr = summary.pop("ratios_best_over_Ustar")
    if rr:
        rr = np.array(rr)
        summary["ratio_best_over_Ustar"] = {
            "n": int(rr.size), "min": float(rr.min()),
            "p25": float(np.percentile(rr, 25)),
            "median": float(np.median(rr)),
            "p75": float(np.percentile(rr, 75)), "max": float(rr.max()),
            "frac_at_optimum": float((rr >= 1.0 - 1e-12).mean())}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--battery", default=None,
                    help="JSON of the matching battery, for comparison")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--labels", default=None,
                    help="subset of labels (comma-separated ids)")
    args = ap.parse_args()

    _reg = np.load(args.register, allow_pickle=True)
    eligible = _reg["eligible"].tolist()
    n = _reg["Y"].shape[0]
    del _reg

    targets = eligible
    if args.labels:
        targets = sorted(int(x) for x in args.labels.split(","))
        assert set(targets) <= set(eligible), "label outside the eligible set"

    t_all = time.time()
    if args.jobs > 1:
        from joblib import Parallel, delayed
        pairs = Parallel(n_jobs=args.jobs)(
            delayed(one_label)(args.register, n, k) for k in targets)
    else:
        pairs = []
        for i, k in enumerate(targets):
            pairs.append(one_label(args.register, n, k))
            r = pairs[-1][1]
            print(f"[{i+1}/{len(targets)}] k={k} R1tau "
                  f"{r['R1_mtau']['pre_verdict']:8s} "
                  f"best={r['R1_mtau']['best_cand']['name']} "
                  f"t={time.time()-t_all:.0f}s", flush=True)

    cells = dict(pairs)
    out = {"meta": {"register": Path(args.register).name,
                    "battery": (Path(args.battery).name
                                if args.battery else None),
                    "candidates": ["C-null", "C-single", "C-full",
                                   "C-greedy"],
                    "ub": "reachable positives per threshold (no solver)",
                    "total_seconds": time.time() - t_all},
           "labels": cells}
    if args.battery:
        out["summary_vs_pli"] = compare_with_battery(cells, args.battery)
        s = out["summary_vs_pli"]
        print(f"SUMMARY cells={s['cells_total']} go_pre={s['go_pre']} "
              f"nogo_pre={s['nogo_pre']} open={s['open']} "
              f"(open: PLI go={s['open_pli_go']} nogo={s['open_pli_nogo']}) "
              f"disagreements={len(s['disagreements'])}", flush=True)

    json.dump(out, open(args.out, "w"), indent=1)
    print(f"DONE {out['meta']['total_seconds']:.0f}s -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
