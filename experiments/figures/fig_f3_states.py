#!/usr/bin/env python3
"""F3: diagnostic-state map, 7 confirmatory datasets x (3 regimes x 2 readings).
Greyscale-safe: state = symbol + cell shading (redundant coding).
States: check = aggregate go without local no-go (all labels viable);
        !+n   = aggregate go with n local no-gos (median across seeds);
        x     = certified global no-go (bound-certified in all seeds);
        ?     = indeterminate (interval [incumbent, upper bound] crosses zero
                within the frozen per-solve budget).
Non-unanimous cells carry a superscript with the count of seeds in the modal state.

The DATA table below is transcribed from the aggregated campaign records
(aggregate_confirmatory / aggregate_gen outputs over the frozen batteries);
this script only renders it.

Usage: fig_f3_states.py [--out OUT.pdf]
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import rcParams

rcParams["font.family"] = "DejaVu Sans"
rcParams["pdf.fonttype"] = 42

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="fig_f3_states.pdf")
args = ap.parse_args()

# (state, n_local_nogo_median or None, seeds_in_modal_state or None if 15/15)
# state in {"ok","cf","ng","ind"}
DATA = {
    "scene":    [("ok",None,None),("ok",None,None),("ok",None,None),("ok",None,None),("ok",None,None),("ok",None,None)],
    "birds":    [("ok",None,None),("ok",None,None),("cf",5,None),("ok",None,13),("cf",1,11),("ok",None,None)],
    "enron":    [("cf",20,None),("cf",4,None),("ng",None,None),("cf",11,None),("ng",None,None),("cf",12,None)],
    "emotions": [("ok",None,None),("ok",None,None),("cf",1,None),("cf",1,12),("cf",1,None),("cf",1,None)],
    "bibtex":   [("cf",10,None),("ok",None,None),("cf",52,None),("cf",1,12),("cf",23,None),("cf",2,None)],
    "yeast":    [("ind",None,None),("cf",5,None),("ng",None,None),("ind",None,None),("ng",None,None),("ind",None,None)],
    "CAL500":   [("ng",None,None),("ind",None,None),("ng",None,None),("ng",None,None),("ng",None,None),("ng",None,None)],
}
ROWS = ["scene","birds","enron","emotions","bibtex","yeast","CAL500"]

SHADE = {"ok": "#ffffff", "cf": "#d9d9d9", "ng": "#8c8c8c", "ind": "#f2f2f2"}
SYMCOL = {"ok": "#000000", "cf": "#000000", "ng": "#ffffff", "ind": "#000000"}

def sym(state, n):
    if state == "ok":  return "✓"
    if state == "cf":  return f"! +{n}"
    if state == "ng":  return "×"
    return "?"

fig, ax = plt.subplots(figsize=(7.0, 3.1))
ncol, nrow = 6, len(ROWS)

for i, base in enumerate(ROWS):
    y = nrow - 1 - i
    for j, (st, n, k) in enumerate(DATA[base]):
        hatch = "///" if st == "ind" else None
        ax.add_patch(Rectangle((j, y), 1, 1, facecolor=SHADE[st],
                               edgecolor="#555555", linewidth=0.6,
                               hatch=hatch))
        label = sym(st, n)
        ax.text(j + 0.47, y + 0.47, label, ha="center", va="center",
                fontsize=11 if st != "cf" else 8.5,
                color=SYMCOL[st],
                fontweight="bold" if st in ("ok", "ng") else "normal")
        if k is not None:
            ax.text(j + 0.92, y + 0.94, f"{k}", ha="right", va="top",
                    fontsize=6.0, color=SYMCOL[st])

# row labels
for i, base in enumerate(ROWS):
    ax.text(-0.15, nrow - 1 - i + 0.5, base, ha="right", va="center",
            fontsize=10, style="italic")

# column headers: regimes grouped, reading sub-headers
regimes = ["R1 (2,2)", "R2 (4,2)", "R3 (2,4)"]
for g, rname in enumerate(regimes):
    ax.text(2*g + 1, nrow + 0.95, rname, ha="center", va="center", fontsize=10.5)
    ax.plot([2*g + 0.06, 2*g + 2 - 0.06], [nrow + 0.62, nrow + 0.62],
            color="#000000", linewidth=0.8)
    for s, rd in enumerate(["0.5", r"$\tau^{*}$"]):
        ax.text(2*g + s + 0.5, nrow + 0.28, rd, ha="center", va="center", fontsize=9.5)

# legend
legend_items = [
    ("ok",  "✓  go, no local no-go"),
    ("cf",  "!+n  go with n local no-gos (median)"),
    ("ng",  "×  certified global no-go"),
    ("ind", "?  indeterminate (budget)"),
]
lx = 6.65
ly = nrow - 0.75
for st, txt in legend_items:
    hatch = "///" if st == "ind" else None
    ax.add_patch(Rectangle((lx, ly), 0.62, 0.62, facecolor=SHADE[st],
                           edgecolor="#555555", linewidth=0.6, hatch=hatch))
    ax.text(lx + 0.31, ly + 0.29, sym(st, "n") if st == "cf" else sym(st, None),
            ha="center", va="center", fontsize=6.5 if st == "cf" else 8.5,
            color=SYMCOL[st],
            fontweight="bold" if st in ("ok", "ng") else "normal")
    ax.text(lx + 0.85, ly + 0.31, txt.split("  ", 1)[1], ha="left", va="center", fontsize=8)
    ly -= 0.95

ax.text(lx, nrow + 0.75, "Aggregate state per cell (15 seeds);\nsuperscript: seeds in modal state\nwhen not unanimous",
        ha="left", va="center", fontsize=7.5, color="#333333")

ax.set_xlim(-1.6, 10.6)
ax.set_ylim(-0.15, nrow + 1.45)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout(pad=0.2)
fig.savefig(args.out, bbox_inches="tight")
print(f"saved {args.out}")
