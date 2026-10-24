"""
Rebuild the six prediction registers at the canonical seed and check them
bit-for-bit against the shipped export (results/registers_seed20260811.npz).

This closes the reproducibility loop: the cross-check track (CPLEX) re-solves
the shipped registers, and this script proves that the shipped registers are
exactly what the environment of record produces from the raw UCI file.

Usage:
    python export_registers.py            # rebuild + compare (default)
    python export_registers.py --write    # rebuild + overwrite the export

Exit code 0 iff every matrix, name list, threshold vector, y and costs match.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from pools_mlp import SEED, POOLS, load, record, cost_structure  # noqa: E402

OUT = ROOT / "results" / f"registers_seed{SEED}.npz"


def rebuild() -> dict[str, np.ndarray]:
    X, y = load(ROOT / "data" / "breast-cancer-wisconsin.data")
    npos, nneg = int(y.sum()), int((1 - y).sum())
    costs, _, _ = cost_structure(npos, nneg)
    payload: dict[str, np.ndarray] = {
        "y": y.astype(np.int8),
        "costs": np.asarray(costs, dtype=float),
        "seed": np.array([SEED]),
    }
    for idx, (label, builder, by_cost) in enumerate(POOLS, start=1):
        M, names, taus = record(builder(SEED), X, y,
                                costs=costs if by_cost else None, seed=SEED)
        payload[f"M_{idx}"] = M.astype(np.int8)
        payload[f"names_{idx}"] = np.asarray(names)
        payload[f"tau_{idx}"] = np.asarray(taus, dtype=float)
        print(f"pool {idx}  m={M.shape[1]:>2}  rebuilt   ({label[:58]})")
    return payload


def main(write: bool) -> int:
    payload = rebuild()
    if write or not OUT.exists():
        np.savez_compressed(OUT, **payload)
        print(f"\nwritten: {OUT}")
        return 0

    shipped = np.load(OUT, allow_pickle=False)
    bad = [k for k in payload
           if k not in shipped.files
           or shipped[k].shape != np.asarray(payload[k]).shape
           or not np.array_equal(np.asarray(shipped[k]).astype(str)
                                 if shipped[k].dtype.kind in "US" else shipped[k],
                                 np.asarray(payload[k]).astype(str)
                                 if np.asarray(payload[k]).dtype.kind in "US"
                                 else np.asarray(payload[k]))]
    if bad:
        print(f"\nMISMATCH against shipped export in: {bad}")
        print("Your environment is NOT the environment of record "
              "(check requirements-highs.txt and the base image digest).")
        return 1
    print("\nEXACT MATCH: rebuilt registers are bit-for-bit identical "
          "to the shipped export.")
    return 0


if __name__ == "__main__":
    sys.exit(main(write="--write" in sys.argv[1:]))
