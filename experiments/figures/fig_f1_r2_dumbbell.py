#!/usr/bin/env python3
"""F1: per-label dumbbell plot of U*_k under regime R2 for the medical
dataset, comparing the fixed 0.5 readout (M^0.5) against the calibrated
readout (M^tau*), with medians and interquartile ranges across the 15
confirmatory seeds. Labels are ordered by prevalence.

Input: the 15 confirmatory battery files battery_conf_seed<SEED>.json
(one per seed; each holds, per label, the audited cells R1/R2/R3 x
{m05, mtau} with the normalized optimum U).

Usage: fig_f1_r2_dumbbell.py [--dir RECORDS_DIR] [--out OUT.pdf]
"""
import argparse
import glob
import json
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="records/confirmatory",
                help="directory with battery_conf_seed*.json (15 seeds)")
ap.add_argument("--out", default="fig_f1_r2_dumbbell.pdf")
args = ap.parse_args()

files = sorted(glob.glob(f"{args.dir}/battery_conf_seed*.json"))
assert len(files) == 15, f"expected 15 battery files, found {len(files)}"

U = defaultdict(lambda: {"m05": [], "mtau": []})
prevalence = {}
for f in files:
    d = json.load(open(f))
    for k, lab in d["labels"].items():
        prevalence[k] = lab["P"] / (lab["P"] + lab["N"])
        for rd in ("m05", "mtau"):
            U[k][rd].append(lab[f"R2_{rd}"]["U"])

labels = sorted(prevalence, key=lambda k: prevalence[k])


def med_iqr(v):
    a = np.asarray(v, dtype=float)
    return (float(np.median(a)), float(np.percentile(a, 25)),
            float(np.percentile(a, 75)))


C_M05, C_MTAU = "#C44E52", "#4C72B0"
plt.rcParams.update({"font.size": 9, "axes.labelsize": 10,
                     "legend.fontsize": 9, "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(7.0, 3.5))

for x, k in enumerate(labels):
    m0, lo0, hi0 = med_iqr(U[k]["m05"])
    mt, lot, hit = med_iqr(U[k]["mtau"])
    ax.plot([x, x], [m0, mt], color="0.6", lw=1.3, zorder=1)
    ax.plot([x, x], [lo0, hi0], color=C_M05, lw=1.1, zorder=2)
    ax.plot([x, x], [lot, hit], color=C_MTAU, lw=1.1, zorder=2)
    ax.plot(x, m0, "o", mfc="white", mec=C_M05, mew=1.4, ms=7, zorder=3)
    ax.plot(x, mt, "o", mfc=C_MTAU, mec=C_MTAU, ms=7, zorder=3)

ax.axhline(0.0, color="black", lw=1.0, zorder=1)
ax.text(len(labels) - 0.6, 2.5, "break-even", ha="right", va="bottom",
        fontsize=9)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([f"{k}\n({prevalence[k]*100:.1f}%)" for k in labels],
                   fontsize=8)
ax.set_xlim(-0.8, len(labels) - 0.2)
ax.set_xlabel("label (prevalence), ordered by prevalence")
ax.set_ylabel(r"$U^{*}_{k}$ under R2")
ax.set_title("global verdict: go in all 15 seeds under both readouts",
             fontsize=9.5, style="italic")
handles = [Line2D([], [], marker="o", mfc="white", mec=C_M05, mew=1.4,
                  ms=7, ls="none", label=r"$M^{0.5}$ (median)"),
           Line2D([], [], marker="o", mfc=C_MTAU, mec=C_MTAU, ms=7,
                  ls="none", label=r"$M^{\tau^{*}}$ (median)")]
ax.legend(handles=handles, loc="lower right", framealpha=0.95)
ax.grid(True, axis="y", ls=":", lw=0.5, color="0.85")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.4)
fig.savefig(args.out, bbox_inches="tight")
print(f"saved {args.out}")
for k in labels:
    m0, lo0, hi0 = med_iqr(U[k]["m05"])
    mt, lot, hit = med_iqr(U[k]["mtau"])
    print(f"label {k:>2} prev {prevalence[k]*100:5.1f}%  "
          f"M0.5 {m0:7.1f} [{lo0:.1f}, {hi0:.1f}]  "
          f"Mtau {mt:7.1f} [{lot:.1f}, {hit:.1f}]")
