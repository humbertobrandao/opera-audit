"""Builds the OOF probabilities of medical (canonical seed).
Per eligible label: 10 stratified folds, 20 configurations, OOF predict_proba.
Output: register_seed<seed>.npz with probs[label] (n x 20), folds, eligible.
"""
import sys, time, hashlib
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
from pool_draft import pool, SEED

HERE = Path(__file__).parent
d = np.load(HERE / "data" / "medical_undivided.npz", allow_pickle=True)
X, Y = d["X"].astype(np.float64), d["Y"].astype(int)
labels = list(d["label_names"])
n, c = Y.shape
K = 10

P = Y.sum(axis=0); N = n - P
eligible = [k for k in range(c) if P[k] >= K and N[k] >= K]
print(f"n={n} c={c} eligible={len(eligible)} (P>=10 and N>=10)", flush=True)

def fit_one(k, j, name, est, tr, te, yk):
    from sklearn.base import clone
    e = clone(est)
    e.fit(X[tr], yk[tr])
    pr = e.predict_proba(X[te])
    cls = list(e.classes_)
    return pr[:, cls.index(1)] if 1 in cls else np.zeros(len(te))

t0 = time.time()
out = {}
for pos, k in enumerate(eligible):
    yk = Y[:, k]
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    folds = [te for _, te in skf.split(X, yk)]
    probs = np.zeros((n, 20), dtype=np.float32)
    jobs = []
    for j, (name, est) in enumerate(pool(SEED)):
        for te in folds:
            tr = np.setdiff1d(np.arange(n), te)
            jobs.append((j, te, delayed(fit_one)(k, j, name, est, tr, te, yk)))
    res = Parallel(n_jobs=2, prefer="processes")(jb for _, _, jb in jobs)
    for (j, te, _), pr in zip(jobs, res):
        probs[te, j] = pr
    out[f"probs_{k}"] = probs
    out[f"folds_{k}"] = np.array([np.pad(f, (0, n - len(f)), constant_values=-1) for f in folds])
    print(f"[{pos+1}/{len(eligible)}] label {k} ({labels[k][:30]}) P={P[k]} "
          f"t={time.time()-t0:.0f}s", flush=True)

out["eligible"] = np.array(eligible)
out["Y"] = Y.astype(np.int8)
np.savez_compressed(HERE / f"register_seed{SEED}.npz", **out)
h = hashlib.sha256((HERE / f"register_seed{SEED}.npz").read_bytes()).hexdigest()
print("DONE", time.time() - t0, "s  sha256:", h, flush=True)
