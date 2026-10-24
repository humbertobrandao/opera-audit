"""Aggregates the pre-certification JSONs into one table per dataset:
how many cells the cheap pre-test closes (go_pre / nogo_pre), how many
stay open and what the ILP adds on them; best-candidate/U* ratio where
the optimum was proved. Output: TSV on stdout + consolidated JSON.

Usage: aggregate_precert.py precert/precert_*.json [--out consolidated.json]
"""
import argparse, json, re, sys
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--out", default=None)
args = ap.parse_args()

rows = defaultdict(lambda: {"cells": 0, "go_pre": 0, "nogo_pre": 0,
                            "open": 0, "open_pli_go": 0, "open_pli_nogo": 0,
                            "open_pli_tl": 0, "disagree": 0,
                            "ratios": [], "t_precert": [], "seeds": set()})

for f in args.files:
    d = json.load(open(f))
    m = re.match(r"precert_(.+)_seed(\d+)\.json", f.split("/")[-1])
    base, seed = (m.group(1), m.group(2)) if m else (f, "?")
    r = rows[base]
    r["seeds"].add(seed)
    s = d.get("summary_vs_pli")
    if s:
        r["cells"] += s["cells_total"]
        r["go_pre"] += s["go_pre"]; r["nogo_pre"] += s["nogo_pre"]
        r["open"] += s["open"]
        r["open_pli_go"] += s["open_pli_go"]
        r["open_pli_nogo"] += s["open_pli_nogo"]
        r["open_pli_tl"] += s.get("pli_status_tl_in_open", 0)
        r["disagree"] += len(s["disagreements"])
        rb = s.get("ratio_best_over_Ustar")
        if rb:
            # rebuilds only the aggregable quantiles: keeps medians
            r["ratios"].append((rb["n"], rb["median"], rb["frac_at_optimum"]))
    for k, rk in d["labels"].items():
        for key, c in rk.items():
            if not isinstance(c, dict) or "pre_verdict" not in c:
                continue
            r["t_precert"].append(c["t_precert"])
            if not s:                      # no battery: count directly
                r["cells"] += 1
                r[c["pre_verdict"] if c["pre_verdict"] != "open"
                  else "open"] += 1

print("base\tseeds\tcells\tgo_pre\tnogo_pre\topen\topen>PLI_go\t"
      "open>PLI_nogo\topen_with_TL\tdisagreements\tratio_median\t"
      "frac_cand_at_optimum\tt_precert_median_ms")
out = {}
for base in sorted(rows):
    r = rows[base]
    tp = np.array(r["t_precert"]) * 1000 if r["t_precert"] else np.array([0.0])
    if r["ratios"]:
        ns = np.array([x[0] for x in r["ratios"]], dtype=float)
        med = float(np.average([x[1] for x in r["ratios"]], weights=ns))
        fat = float(np.average([x[2] for x in r["ratios"]], weights=ns))
    else:
        med = fat = float("nan")
    print(f"{base}\t{len(r['seeds'])}\t{r['cells']}\t{r['go_pre']}\t"
          f"{r['nogo_pre']}\t{r['open']}\t{r['open_pli_go']}\t"
          f"{r['open_pli_nogo']}\t{r['open_pli_tl']}\t{r['disagree']}\t"
          f"{med:.3f}\t{fat:.3f}\t{np.median(tp):.1f}")
    out[base] = {k: (sorted(v) if isinstance(v, set) else v)
                 for k, v in r.items() if k not in ("ratios", "t_precert")}
    out[base]["ratio_median_wavg"] = med
    out[base]["frac_at_optimum_wavg"] = fat
    out[base]["t_precert_ms_median"] = float(np.median(tp))

if args.out:
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"-> {args.out}", file=sys.stderr)
