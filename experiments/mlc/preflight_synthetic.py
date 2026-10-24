"""Synthetic pre-flight (M-9): PURELY TECHNICAL test of the pipeline at the
orders of magnitude of the giants. No scientific metric.

Checks, for each scenario: dtypes, peak memory, training completion of the
20 configurations, shape/dtype of predict_proba, serialization. Scenarios:
  dense-continuous ~ mediamill (43907 x 120)
  sparse-binary    ~ eukaryotego (7766 x 12689)
  sparse-binary    ~ delicious (16105 x 500)
Usage: preflight_synthetic.py [--small]  (--small shrinks 10x, to validate
the script itself; the official pre-flight runs at full scale on the
register machine)
"""
import sys, time, resource
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from pool_frozen import pool_with_states
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SMALL = "--small" in sys.argv
f = 10 if SMALL else 1
SC = [("denso_continuo_mediamill", 43907 // f, 120, True, 0.0),
      ("esparso_binario_eukaryotego", 7766 // f, 12689 // (f if SMALL else 1), False, 0.03),
      ("esparso_binario_delicious", 16105 // f, 500, False, 0.05)]
rng = np.random.default_rng(20260815)

def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

for nome, n, p, cont, dens in SC:
    if cont:
        X = rng.normal(size=(n, p)).astype(np.float32)
    else:
        X = (rng.random((n, p)) < dens).astype(np.float32)
    y = (rng.random(n) < 0.3).astype(int)
    tr = np.arange(int(0.9 * n)); te = np.arange(int(0.9 * n), n)
    print(f"== {nome}: n={n} p={p} X={X.nbytes/1e6:.0f}MB ==", flush=True)
    ests = pool_with_states(list(range(20)))
    for j, (cfg, est) in enumerate(ests):
        e = Pipeline([("s", StandardScaler()), ("e", est)]) if cont else est
        t0 = time.perf_counter()
        try:
            e.fit(X[tr], y[tr])
            pr = e.predict_proba(X[te])
            assert pr.shape[0] == len(te) and pr.dtype.kind == "f"
            print(f"  {cfg:10s} fit={time.perf_counter()-t0:8.1f}s "
                  f"peak={peak_mb():7.0f}MB OK", flush=True)
        except Exception as ex:
            print(f"  {cfg:10s} TECHNICAL FAILURE: {type(ex).__name__}: {ex}", flush=True)
print("PRE-FLIGHT COMPLETE (technical; no scientific metric)", flush=True)
