"""
Fixed voting rules on the EXACT register of Audit 3 (rebuilt pool, tau*).

Evaluates, on the shipped register results/registers_seed20260811.npz and
under the published cost structure, the fixed rules of the paper's
consensus remark: best single classifier, OR (t=1), strict majority
(t = floor(m/2)+1) and AND (t=m) over the full pool, next to the certified
optimum. Every rule is a decision of the audited family, hence bounded by
U* (oracle-bound corollary). Deterministic post-processing: no training,
no solving, no randomness.

Usage:
    python baselines_table.py

Expected output (paper values):
    best single      |J|=1 t=1  FN=0 FP=47   U=21477
    OR               |J|=5 t=1  FN=0 FP=233  U=10131
    strict majority  |J|=5 t=3  FN=0 FP=47   U=21477
    AND              |J|=5 t=5  FN=1 FP=22   U=-3098
    OPERA optimum                             U=21477
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "results" / "registers_seed20260811.npz"
POOL = "4"                 # rebuilt pool: five well-fitted MLPs read at tau*
U_STAR = 21477.0           # published optimum of Audit 3

EXPECTED = {
    "best single":     (21477.0, 0, 47),
    "OR":              (10131.0, 0, 233),
    "strict majority": (21477.0, 0, 47),
    "AND":             (-3098.0, 1, 22),
}


def utility(M: np.ndarray, y: np.ndarray, costs: np.ndarray,
            cols: np.ndarray, t: int) -> tuple[float, int, int]:
    r_tp, r_tn, r_fn, r_fp = costs
    z = (M[:, cols].sum(axis=1) >= t).astype(int)
    tp = int(((z == 1) & (y == 1)).sum())
    fp = int(((z == 1) & (y == 0)).sum())
    fn = int(((z == 0) & (y == 1)).sum())
    tn = int(((z == 0) & (y == 0)).sum())
    return float(r_tp * tp + r_tn * tn + r_fn * fn + r_fp * fp), fn, fp


def main() -> int:
    data = np.load(NPZ, allow_pickle=False)
    y = data["y"].astype(int)
    costs = data["costs"]
    M = data[f"M_{POOL}"].astype(int)
    m = M.shape[1]
    every = np.arange(m)

    singles = [utility(M, y, costs, np.array([j]), 1) for j in range(m)]
    rows = {
        "best single":     (max(singles), 1, 1),
        "OR":              (utility(M, y, costs, every, 1), m, 1),
        "strict majority": (utility(M, y, costs, every, m // 2 + 1),
                            m, m // 2 + 1),
        "AND":             (utility(M, y, costs, every, m), m, m),
    }

    print(f"register: pool {POOL} (rebuilt, tau*)  n={len(y)}  m={m}")
    print(f"{'rule':<16} {'|J|':>4} {'t':>3} {'FN':>4} {'FP':>4} {'U':>10}  check")
    failures = 0
    for name, ((U, fn, fp), J, t) in rows.items():
        exp = EXPECTED[name]
        ok = (abs(U - exp[0]) < 1e-9 and fn == exp[1] and fp == exp[2]
              and U <= U_STAR + 1e-9)
        failures += 0 if ok else 1
        print(f"{name:<16} {J:>4} {t:>3} {fn:>4} {fp:>4} {U:>10.0f}  "
              f"{'ok' if ok else f'DIVERGES from {exp}'}")
    print(f"{'OPERA optimum':<16} {'':>4} {'':>3} {'':>4} {'':>4} "
          f"{U_STAR:>10.0f}  (certified U*)")
    print("\n" + ("ALL VALUES MATCH THE PUBLISHED TABLE."
                  if failures == 0 else f"{failures} DIVERGENCE(S)."))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
