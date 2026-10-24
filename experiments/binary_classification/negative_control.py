"""
Negative control by label permutation (supplementary material).

PROTOCOL, DECLARED BEFORE EXECUTION (2026-08-20):
  For each of the three matrices reported in the paper (initial 0.5,
  enlarged 0.5, rebuilt tau*), all frozen in
  results/registers_seed20260811.npz:
    1. keep M fixed;
    2. draw B = 200 independent label permutations of y, each preserving
       prevalence exactly (|I+| = 239, |I-| = 444), with NumPy RNG seeded
       at 20260811;
    3. solve OPERA (HiGHS backend, zero relative gap) on (M, y_perm) with
       the published cost structure, unchanged (conditions (D) and (T)
       depend only on prevalence, hence still hold);
    4. report, per matrix: the fraction of permutations with U* > 0
       (spurious go rate), the maximum and median U*_perm, and the
       fraction with FN = 0 at the optimum.
  The result is reported regardless of direction. It is a negative
  control of THESE instances and THIS cost structure: it does not
  estimate generalization, a universal false-go rate, or statistical
  significance.

Usage:
    python negative_control.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from opera.model import Instance, solve  # noqa: E402

B = 200
RNG_SEED = 20260811
POOLS = {"3": "initial (0.5)", "6": "enlarged (0.5)", "4": "rebuilt (tau*)"}


def main() -> int:
    data = np.load(ROOT / "results" / "registers_seed20260811.npz",
                   allow_pickle=False)
    y = data["y"].astype(int)
    costs = data["costs"]
    rng = np.random.default_rng(RNG_SEED)

    print(f"negative control: B={B} prevalence-preserving permutations "
          f"per matrix, RNG seed {RNG_SEED}")
    print(f"history: n={len(y)}  |I+|={int(y.sum())}  |I-|={int((1-y).sum())}")
    print(f"{'pool':>8} {'go rate':>9} {'max U*':>12} {'median U*':>12} "
          f"{'FN=0 rate':>10}")
    for key, label in POOLS.items():
        M = data[f"M_{key}"].astype(int)
        us, fn0 = [], 0
        for _ in range(B):
            yp = rng.permutation(y)
            sol = solve(Instance(M, yp, costs[None, :]), backend="highs")
            us.append(sol.U_star)
            fn0 += 1 if sol.confusion[0]["fn"] == 0 else 0
        us = np.array(us)
        print(f"{key+' '+label:>22} {float((us > 0).mean()):>9.3f} "
              f"{us.max():>12.0f} {np.median(us):>12.0f} {fn0/B:>10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
