"""Aggregation of the 15 confirmatory seeds (protocol v3, Section 10).
The SEED is the replicate unit: statistics per seed, then median/IQR/
min-max across seeds. Also produces the data of F1 (medians and IQR of
U*_k per label across seeds).
Usage: aggregate_confirmatory.py OUT.json battery1.json battery2.json ...
"""
import json, sys
import numpy as np

out_path = sys.argv[1]
files = sys.argv[2:]
bats = [json.load(open(f))["labels"] for f in files]
S = len(bats)
labels = sorted(bats[0].keys(), key=int)
REG = ("R1", "R2", "R3")
RD = ("m05", "mtau")

def iqr(v):
    return float(np.percentile(v, 75) - np.percentile(v, 25))

per_seed = []
for si, b in enumerate(bats):
    row = {"seed_index": si, "file": files[si]}
    for r in REG:
        for rd in RD:
            cs = [b[k][f"{r}_{rd}"] for k in labels]
            Us = [c["U"] for c in cs]
            row[f"{r}_{rd}"] = {
                "U_bar": float(np.mean(Us)),
                "U_sum": float(np.sum(Us)),
                "GO": int(sum(c["verdict"] == "go" for c in cs)),
                "local_nogo": int(sum(c["verdict"] != "go" for c in cs)),
                "dU_bar": float(np.mean([c["dU"] for c in cs])),
                "gcons_pos": int(sum(c["gcons_int"] > 0 for c in cs)),
                "global_verdict": "go" if np.sum(Us) > 0 else "no-go",
            }
        row[f"{r}_flips"] = int(sum(
            b[k][f"{r}_m05"]["verdict"] != b[k][f"{r}_mtau"]["verdict"]
            for k in labels))
        # RQ1: global GO with at least one local NO-GO (per readout)
        for rd in RD:
            g = row[f"{r}_{rd}"]
            g["rq1_hit"] = bool(g["global_verdict"] == "go"
                                and g["local_nogo"] >= 1)
    per_seed.append(row)

summary = {}
for r in REG:
    for rd in RD:
        key = f"{r}_{rd}"
        for f in ("U_bar", "GO", "local_nogo", "dU_bar", "gcons_pos"):
            v = [ps[key][f] for ps in per_seed]
            summary[f"{key}.{f}"] = {
                "median": float(np.median(v)), "iqr": iqr(v),
                "min": float(np.min(v)), "max": float(np.max(v))}
        summary[f"{key}.rq1_freq"] = int(sum(ps[key]["rq1_hit"]
                                             for ps in per_seed))
        summary[f"{key}.global_go_freq"] = int(sum(
            ps[key]["global_verdict"] == "go" for ps in per_seed))
    v = [ps[f"{r}_flips"] for ps in per_seed]
    summary[f"{r}.flips"] = {"median": float(np.median(v)), "iqr": iqr(v),
                             "min": int(np.min(v)), "max": int(np.max(v))}

# flip and NO-GO frequency per label across seeds
label_stats = {}
for k in labels:
    ls = {}
    for r in REG:
        ls[f"{r}_flip_freq"] = int(sum(
            b[k][f"{r}_m05"]["verdict"] != b[k][f"{r}_mtau"]["verdict"]
            for b in bats))
        for rd in RD:
            ls[f"{r}_{rd}_nogo_freq"] = int(sum(
                b[k][f"{r}_{rd}"]["verdict"] != "go" for b in bats))
            Us = [b[k][f"{r}_{rd}"]["U"] for b in bats]
            ls[f"{r}_{rd}_U_med"] = float(np.median(Us))
            ls[f"{r}_{rd}_U_iqr"] = iqr(Us)
    # frontier (tau* readouts)
    frs = [b[k]["R1_mtau"].get("frontier", {}) for b in bats]
    fin = [f.get("rho") for f in frs if f.get("case") == "finito"]
    ls["R1_frontier_finite"] = len(fin)
    ls["R1_frontier_inf"] = sum(1 for f in frs if str(f.get("case", "")).startswith("inf"))
    if fin:
        ls["R1_rho_med"] = float(np.median(fin))
    label_stats[k] = ls

json.dump({"n_seeds": S, "per_seed": per_seed, "summary": summary,
           "per_label": label_stats},
          open(out_path, "w"), indent=1)

print(f"=== {S} confirmatory seeds ===")
for r in REG:
    for rd in RD:
        k = f"{r}_{rd}"
        s = summary
        print(f"{k}: Ubar med={s[k+'.U_bar']['median']:7.2f} "
              f"(IQR {s[k+'.U_bar']['iqr']:5.2f}, "
              f"min {s[k+'.U_bar']['min']:7.2f}) | "
              f"GO med={s[k+'.GO']['median']:4.1f}/21 "
              f"(min {s[k+'.GO']['min']:.0f}) | "
              f"NO-GO local med={s[k+'.local_nogo']['median']:.1f} | "
              f"Gcons>0 med={s[k+'.gcons_pos']['median']:.0f}/21 | "
              f"RQ1 {s[k+'.rq1_freq']}/{S} | "
              f"GO global {s[k+'.global_go_freq']}/{S}")
    print(f"{r}: flips 0.5->tau* med={summary[r+'.flips']['median']:.0f} "
          f"(min {summary[r+'.flips']['min']}, max {summary[r+'.flips']['max']})")
fl = [(k, sum(label_stats[k][f"{r}_flip_freq"] for r in REG)) for k in labels]
fl = [x for x in sorted(fl, key=lambda t: -t[1]) if x[1] > 0]
print("labels flipping in some seed/regime:",
      ", ".join(f"k={k}({c})" for k, c in fl[:12]) if fl else "none")
print("DONE ->", out_path)
