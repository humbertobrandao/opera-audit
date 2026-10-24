"""
Solver cross-validation: HiGHS vs CPLEX on the EXACT registers of the paper.

Loads results/registers_seed20260811.npz — the prediction matrices that
produced every number reported in Section 6 (canonical seed, exported from
the environment of record) — and audits each one with both backends.

Because both solvers receive identical input, any disagreement in U*, FN or
FP is a solver/implementation issue, never a data or library-version issue.
The optimal decision d* (members, threshold) MAY differ between solvers when
alternative optima exist; that is expected and not a failure. The equality
criterion is: U*, and the confusion matrix implied by each solver's own d*
being consistent with its U*.

Usage:
    python validation/compare_solvers.py            # both backends
    python validation/compare_solvers.py highs      # one backend only
    python validation/compare_solvers.py cplex

Exit code 0 iff every U* matches the published value and, when both backends
run, they agree with each other.
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opera.model import Instance, solve  # noqa: E402

# Published values (Section 6, seed 20260811). Keys = pool ids in the npz.
PUBLISHED = {
    "1": dict(U=-32004.0, fn=2, fp=68),
    "2": dict(U=-2679.0,  fn=0, fp=443),
    "3": dict(U=-81032.0, fn=4, fp=16),    # paper: initial
    "4": dict(U=21477.0,  fn=0, fp=47),    # paper: rebuilt
    "5": dict(U=20623.0,  fn=0, fp=61),
    "6": dict(U=-28771.0, fn=2, fp=15),    # paper: enlarged
}
PAPER_LABEL = {"3": "initial", "4": "rebuilt", "6": "enlarged"}


def versions() -> str:
    import scipy
    lines = [f"python {sys.version.split()[0]} | {platform.platform()}",
             f"numpy {np.__version__} | scipy {scipy.__version__}"]
    try:
        import scipy.optimize._highspy._core as hc
        lines.append(f"highs {hc.HIGHS_VERSION_MAJOR}.{hc.HIGHS_VERSION_MINOR}."
                     f"{hc.HIGHS_VERSION_PATCH} (bundled in scipy)")
    except Exception:
        lines.append("highs: version not exposed by this scipy")
    try:
        import docplex
        import cplex
        lines.append(f"docplex {docplex.__version__} | cplex {cplex.__version__}")
    except ImportError as e:
        lines.append(f"cplex/docplex not importable: {e}")
    return "\n".join(lines)


def main(backends):
    print(versions(), "\n")
    data = np.load(ROOT / "results" / "registers_seed20260811.npz",
                   allow_pickle=False)
    y, costs = data["y"], data["costs"]
    print(f"history: n={len(y)}  |I+|={int(y.sum())}  |I-|={int((1-y).sum())}")
    print(f"costs: r_tp={costs[0]:.0f} r_tn={costs[1]:.0f} "
          f"r_fn={costs[2]:.0f} r_fp={costs[3]:.0f}\n")

    header = f"{'pool':>5} {'backend':>7} {'U*':>10} {'FN':>4} {'FP':>4} " \
             f"{'|J|':>4} {'t':>3} {'s':>8}   check"
    print(header); print("-" * len(header))

    failures = 0
    for key in sorted(PUBLISHED):
        M = data[f"M_{key}"]
        got = {}
        for backend in backends:
            sol = solve(Instance(M, y, costs[None, :]), backend=backend)
            cm = sol.confusion[0]
            got[backend] = (sol.U_star, cm["fn"], cm["fp"])
            pub = PUBLISHED[key]
            ok_pub = (abs(sol.U_star - pub["U"]) < 1e-6
                      and cm["fn"] == pub["fn"] and cm["fp"] == pub["fp"])
            # d* may legitimately differ between solvers; U*/FN/FP must not.
            mark = "ok" if ok_pub else f"DIVERGES from published {pub}"
            failures += 0 if ok_pub else 1
            tag = PAPER_LABEL.get(key, "")
            print(f"{key+('*' if tag else ''):>5} {backend:>7} "
                  f"{sol.U_star:>10.0f} {cm['fn']:>4} {cm['fp']:>4} "
                  f"{len(sol.selected[0]):>4} {sol.thresholds[0]:>3} "
                  f"{sol.seconds:>8.3f}   {mark} {('('+tag+')') if tag else ''}")
        if len(backends) == 2:
            same = all(abs(a - b) < 1e-6 if isinstance(a, float) else a == b
                       for a, b in zip(got[backends[0]], got[backends[1]]))
            if not same:
                failures += 1
                print(f"      >>> BACKENDS DISAGREE on pool {key}: {got}")

    print("\n" + ("ALL VALUES MATCH THE PUBLISHED RESULTS." if failures == 0
                  else f"{failures} DIVERGENCE(S). Investigate before use."))
    return 1 if failures else 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main([arg] if arg in ("highs", "cplex") else ["highs", "cplex"]))
