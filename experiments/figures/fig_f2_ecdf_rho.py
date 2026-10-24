#!/usr/bin/env python3
"""F2: ECDF of the finite cost-ratio frontiers rho*_k per regime (R1-R3)
under the calibrated readout (M^tau*), pooled over the 21 labels x 15
confirmatory seeds of the medical dataset (315 label-cells per regime).
The legend reports, per regime, how many of the 315 frontiers are finite
and the complementary fraction of infinite frontiers.

Frontier record semantics (field values are frozen in the archived JSONs):
  frontier.case == "finito" -> finite frontier, frontier.rho holds rho*_k;
  frontier.case == "inf"    -> infinite frontier (go for every rho > 0).

Input: the 15 confirmatory battery files battery_conf_seed<SEED>.json.

Usage: fig_f2_ecdf_rho.py [--dir RECORDS_DIR] [--out OUT.pdf]
"""
import argparse
import glob
import json
import statistics as st

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="records/confirmatory",
                help="directory with battery_conf_seed*.json (15 seeds)")
ap.add_argument("--out", default="fig_f2_ecdf_rho.pdf")
args = ap.parse_args()

files = sorted(glob.glob(f"{args.dir}/battery_conf_seed*.json"))
assert len(files) == 15, f"expected 15 battery files, found {len(files)}"

REGIMES = [("R1", "#4C72B0", "-"),
           ("R2", "#C44E52", "--"),
           ("R3", "#55A868", "-.")]

rhos = {r: [] for r, _, _ in REGIMES}
total = {r: 0 for r, _, _ in REGIMES}
for f in files:
    d = json.load(open(f))
    for lab in d["labels"].values():
        for reg, _, _ in REGIMES:
            fr = lab[f"{reg}_mtau"].get("frontier")
            total[reg] += 1
            if fr and fr.get("case") == "finito" and fr.get("rho") is not None:
                rhos[reg].append(float(fr["rho"]))

plt.rcParams.update({"font.size": 9, "axes.labelsize": 10,
                     "legend.fontsize": 9, "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(7.0, 2.9))

for reg, col, ls in REGIMES:
    v = np.sort(np.asarray(rhos[reg]))
    n, N = len(v), total[reg]
    pct_inf = round(100 * (N - n) / N)
    y = np.arange(1, n + 1) / n
    ax.step(v, y, where="post", color=col, ls=ls, lw=1.6,
            label=f"{reg}: {n}/{N} finite ({pct_inf}% infinite)")
    print(f"{reg}: finite {n}/{N}  median {st.median(v):.1f}  "
          f"min {v[0]:.1f}  max {v[-1]:.1f}")

ax.axvline(1.0, color="0.4", lw=0.9, ls=":")
ax.text(1.12, 0.05, r"$\rho = 1$ (nominal)", fontsize=9, color="0.2")
ax.set_xscale("log")
ax.set_ylim(-0.02, 1.04)
ax.set_xlabel(r"frontier $\rho^{*}_{k}$ (log scale)")
ax.set_ylabel("cumulative fraction (finite)")
ax.legend(loc="upper left", framealpha=0.95)
ax.grid(True, which="major", ls=":", lw=0.5, color="0.85")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.4)
fig.savefig(args.out, bbox_inches="tight")
print(f"saved {args.out}")
