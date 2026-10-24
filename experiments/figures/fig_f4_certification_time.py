#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F4: fraction of re-measured sample cells with a certified verdict and
with a proved optimum up to time t (censored at the per-solve budget).
Data: the timing batteries measured on the dedicated workstation, already
archived. Style matches figures F2/F3 (English labels, step curves, log axis).

Usage: fig_f4_certification_time.py [--dir RECORDS_DIR] [--out OUT.pdf]
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="records/timing",
                help="directory with timing_battery_<dataset>_seed*.json")
ap.add_argument("--out", default="fig_f4_certification_time.pdf")
args = ap.parse_args()

BASES = [("eukaryotego", "EukaryoteGO", "#0072B2"),
         ("mediamill",   "Mediamill",   "#D55E00"),
         ("delicious",   "Delicious",   "#009E73")]
TL = 600.0
T0 = 5e-3   # plotting floor for events "before the first callback" (None) and near-zero times

def cells_of(base):
    d = json.load(open(f"{args.dir}/timing_battery_{base}_seed1563301818.json"))
    return [v for c in d["labels"].values() for v in c.values()
            if isinstance(v, dict) and "t_solver" in v]

def is_opt(c):  return str(c.get("status", "")).startswith("optimal")
def indet(c):   return (not is_opt(c)) and c["U_int"] <= 0 and c["root_bound"] > 0

def ecdf(times, n, grid):
    t = np.sort(np.asarray(times, dtype=float))
    return np.searchsorted(t, grid, side="right") / n

grid = np.logspace(np.log10(T0), np.log10(TL), 600)
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9,
                     "legend.fontsize": 7.6, "axes.linewidth": 0.7,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
fig, ax = plt.subplots(figsize=(7.0, 2.85))

summary = []
for key, name, col in BASES:
    cs = cells_of(key); n = len(cs)
    tv = [max(T0, c["T_verdict"] if c["T_verdict"] is not None else 0.0)
          for c in cs if not indet(c)]
    tp = [max(T0, c["t_solver"]) for c in cs if is_opt(c)]
    ax.plot(grid, ecdf(tv, n, grid), color=col, lw=1.4, ls="-", label=name)
    ax.plot(grid, ecdf(tp, n, grid), color=col, lw=1.2, ls=(0, (4, 2)))
    summary.append((name, n, len(tv), len(tp),
                    round(100*len(tv)/n, 1), round(100*len(tp)/n, 1)))

ax.axvline(TL, color="0.35", lw=0.8, ls=":")
ax.text(TL*0.88, 0.035, "budget (600 s)", rotation=90, fontsize=7,
        color="0.25", ha="right", va="bottom")
ax.set_xscale("log")
ax.set_xlim(T0, TL*1.35)
ax.set_ylim(0, 1.02)
ax.set_xlabel("time $t$ (s, log scale)")
ax.set_ylabel("fraction of sample cells $\\leq t$")
ax.grid(True, which="major", ls=":", lw=0.5, color="0.8")
from matplotlib.lines import Line2D
handles, labels = ax.get_legend_handles_labels()
handles += [Line2D([], [], color="0.2", lw=1.4, ls="-"),
            Line2D([], [], color="0.2", lw=1.2, ls=(0, (4, 2)))]
labels += ["certified verdict", "optimality proof"]
ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.01),
          ncol=5, framealpha=0.0, borderpad=0.2, handlelength=2.2,
          columnspacing=1.2, edgecolor="none")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.4)
fig.savefig(args.out)
print(f"saved {args.out}")
for r in summary:
    print(f"{r[0]:12s} n={r[1]} cert={r[2]} ({r[4]}%)  proofs={r[3]} ({r[5]}%)")
