"""Frozen aggregator of the generality/scale study (protocol sha256 304b2e52).
Seed = replicate unit. F-A with 4 states; flips in both directions;
q^cons/q^AMAJ/q^B per seed; rho~_s and p^inf_s per seed; prevalence rank
of the flips aggregated per seed (average rank on ties; '--' when there
are no flips).
Usage: aggregate_gen.py BASE OUT.json battery1.json battery2.json ...
"""
import sys, json
import numpy as np

BASE, OUTF = sys.argv[1], sys.argv[2]
bats = [json.load(open(f))["labels"] for f in sys.argv[3:]]
S = len(bats)
labels = sorted(bats[0].keys(), key=int)
L = len(labels)
REG = ("R1", "R2", "R3"); RD = ("m05", "mtau")
P = {k: bats[0][k]["P"] for k in labels}
n = bats[0][labels[0]]["P"] + bats[0][labels[0]]["N"]

# normalized prevalence rank (average rank on ties), over the eligible labels
from scipy.stats import rankdata
pv = np.array([P[k] for k in labels], float)
rk = rankdata(pv, method="average")
qk = {k: (rk[i] - 1) / (L - 1) if L > 1 else float("nan")
      for i, k in enumerate(labels)}

def iqr(v): return float(np.percentile(v, 75) - np.percentile(v, 25))

per_seed = []
for b in bats:
    row = {}
    for r in REG:
        for rd in RD:
            cs = [b[k][f"{r}_{rd}"] for k in labels]
            Us = [c["U"] for c in cs]
            glob = float(np.sum(Us))
            locs_nogo = sum(c["verdict"] != "go" for c in cs)
            if glob <= 0:
                fa = "global_nogo"
            elif locs_nogo == 0:
                fa = "go_sem_conflito"
            else:
                fa = "conflito"
            row[f"{r}_{rd}"] = {
                "U_bar": float(np.mean(Us)), "U_sum": glob,
                "estado_FA": fa, "local_nogo": int(locs_nogo),
                "q_cons": sum(c["gcons_int"] > 0 for c in cs) / L,
                "q_AMAJ": sum(c["gap_AMAJ_int"] > 0 for c in cs) / L,
                "q_B": sum(c["gap_B_int"] > 0 for c in cs) / L}
        # flips in both directions + prevalence rank per direction
        fpos = [k for k in labels if b[k][f"{r}_m05"]["verdict"] != "go"
                and b[k][f"{r}_mtau"]["verdict"] == "go"]
        fneg = [k for k in labels if b[k][f"{r}_m05"]["verdict"] == "go"
                and b[k][f"{r}_mtau"]["verdict"] != "go"]
        row[f"{r}_flips"] = {"pos": len(fpos), "neg": len(fneg),
                             "rank_pos": float(np.median([qk[k] for k in fpos])) if fpos else None,
                             "rank_neg": float(np.median([qk[k] for k in fneg])) if fneg else None}
    # frontier (tau*), per seed
    for r in REG:
        fin, inf_ct = [], 0
        for k in labels:
            fr = b[k][f"{r}_mtau"].get("frontier", {})
            c = str(fr.get("case", ""))
            if c == "finito": fin.append(fr["rho"])
            elif c.startswith("inf"): inf_ct += 1
        row[f"{r}_rho"] = {"rho_med": float(np.median(fin)) if fin else None,
                           "p_inf": inf_ct / L, "n_fin": len(fin)}
    per_seed.append(row)

def med_mm(v):
    v = [x for x in v if x is not None]
    if not v: return None
    return {"med": float(np.median(v)), "min": float(np.min(v)),
            "max": float(np.max(v)), "iqr": iqr(v)}

summary = {"base": BASE, "n": n, "L_eleg": L, "n_seeds": S}
for r in REG:
    for rd in RD:
        key = f"{r}_{rd}"
        st = [ps[key]["estado_FA"] for ps in per_seed]
        summary[key] = {
            "estados_FA": {s0: st.count(s0) for s0 in
                           ("global_nogo", "go_sem_conflito", "conflito")},
            "U_bar": med_mm([ps[key]["U_bar"] for ps in per_seed]),
            "local_nogo": med_mm([ps[key]["local_nogo"] for ps in per_seed]),
            "q_cons": med_mm([ps[key]["q_cons"] for ps in per_seed]),
            "q_AMAJ": med_mm([ps[key]["q_AMAJ"] for ps in per_seed]),
            "q_B": med_mm([ps[key]["q_B"] for ps in per_seed])}
    summary[f"{r}_flips_pos"] = med_mm([ps[f"{r}_flips"]["pos"] for ps in per_seed])
    summary[f"{r}_flips_neg"] = med_mm([ps[f"{r}_flips"]["neg"] for ps in per_seed])
    summary[f"{r}_rank_pos"] = med_mm([ps[f"{r}_flips"]["rank_pos"] for ps in per_seed])
    summary[f"{r}_rho_med"] = med_mm([ps[f"{r}_rho"]["rho_med"] for ps in per_seed])
    summary[f"{r}_p_inf"] = med_mm([ps[f"{r}_rho"]["p_inf"] for ps in per_seed])

json.dump({"summary": summary, "per_seed": per_seed, "rank_prevalencia": qk},
          open(OUTF, "w"), indent=1)

print(f"=== {BASE}: {S} seeds, L_eleg={L} ===")
for r in REG:
    for rd in RD:
        s = summary[f"{r}_{rd}"]
        e = s["estados_FA"]
        print(f"{r}_{rd}: FA[go_sem_confl={e['go_sem_conflito']} conflito={e['conflito']} "
              f"global_nogo={e['global_nogo']}]/{S}  Ubar_med={s['U_bar']['med']:7.2f}  "
              f"q_cons_med={s['q_cons']['med']:.2f}")
    fp_ = summary[f"{r}_flips_pos"]; fn_ = summary[f"{r}_flips_neg"]
    print(f"{r}: flips no-go->go med={fp_['med']:.0f} [{fp_['min']:.0f}-{fp_['max']:.0f}]  "
          f"go->no-go med={fn_['med']:.0f} [{fn_['min']:.0f}-{fn_['max']:.0f}]")
print("DONE ->", OUTF)
