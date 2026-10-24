"""
Stability of the verdicts under re-drawing.

The design of the six pools was fixed BEFORE any audit was run. The question
here is not "is there a seed on which the story works" — that would be worth
nothing — but "how often does each verdict repeat when the whole pipeline is
re-drawn". The counts are reported as measured, including where they run
against the intended narrative.

Re-drawing the split is not cross-validation: the audit protocol stays oracle
over 100% of the history. What is being tested is the robustness of the
CONCLUSION.

Output: results/bcw_seeds.json
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from opera.model import Instance, solve                        # noqa: E402
from pools_mlp import POOLS, load, record, cost_structure      # noqa: E402

SEEDS = [20260811, 7, 101, 2718, 31415, 4242, 555, 90210,
         13, 271828, 1618, 6022, 8080, 4711, 99991]
DATA = ROOT / "data" / "breast-cancer-wisconsin.data"


def one_seed(sd: int):
    X, yh = load(DATA)
    npos, nneg = int(yh.sum()), int((1 - yh).sum())
    costs, _, _ = cost_structure(npos, nneg)
    out = []
    for label, builder, by_cost in POOLS:
        M, names, taus = record(builder(sd), X, yh,
                                costs=costs if by_cost else None, seed=sd)
        sol = solve(Instance(M, yh, costs[None, :]), backend="highs")
        cm = sol.confusion[0]
        fn_union = int((((M.sum(axis=1) >= 1) == 0) & (yh == 1)).sum())
        out.append(dict(pool=label, m=M.shape[1], verdict=sol.verdict,
                        U=sol.U_star, t=sol.thresholds[0],
                        J=len(sol.selected[0]), fn_union=fn_union, **cm))
    return out


if __name__ == "__main__":
    everything, counts = {}, {p[0]: Counter() for p in POOLS}
    for sd in SEEDS:
        rows = one_seed(sd)
        everything[sd] = rows
        print(f"seed {sd}:")
        for r in rows:
            counts[r["pool"]][r["verdict"]] += 1
            print(f"   {r['pool'][:52]:<54} {r['verdict']:>5}  "
                  f"U*={r['U']:>10.0f}  |J|={r['J']} t={r['t']}  "
                  f"FN={r['fn']:>3} FP={r['fp']:>3}  (FN of union={r['fn_union']})")

    print(f"\n=== verdict frequency over {len(SEEDS)} seeds ===")
    for label, _, _ in POOLS:
        c = counts[label]
        print(f"  {label[:52]:<54} no-go={c['no-go']:>3}  go={c['go']:>3}")

    out = ROOT / "results"; out.mkdir(exist_ok=True)
    (out / "bcw_seeds.json").write_text(json.dumps(
        {"seeds": SEEDS, "results": {str(k): v for k, v in everything.items()},
         "counts": {k: dict(v) for k, v in counts.items()}},
        indent=2, ensure_ascii=False))
    print(f"\nwritten to {out / 'bcw_seeds.json'}")
