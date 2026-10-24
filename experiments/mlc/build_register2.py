"""CONFIRMATORY OOF register builder (protocol v3, Sections 3-4).
Deterministic derivation: folds per label via SeedSequence([seed, k,
0xF01D5]); random_state of configuration j via SeedSequence([seed, k, j]).
Usage: build_register2.py SEED [N_JOBS]
"""
import sys, time, hashlib
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
from pool_frozen import pool_with_states

SEED = int(sys.argv[1])
NJOBS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
K = 10
HERE = Path(__file__).parent
d = np.load(HERE / "data" / "medical_undivided.npz", allow_pickle=True)
X, Y = d["X"].astype(np.float64), d["Y"].astype(int)
labels = list(d["label_names"])
n, c = Y.shape
P = Y.sum(axis=0); N = n - P
eligible = [k for k in range(c) if P[k] >= K and N[k] >= K]
print(f"seed={SEED} n={n} c={c} eligible={len(eligible)}", flush=True)

def state(*args):
    return int(np.random.SeedSequence(list(args)).generate_state(1)[0] % (2**31))

def fit_one(est, tr, te, yk):
    from sklearn.base import clone
    e = clone(est); e.fit(X[tr], yk[tr])
    pr = e.predict_proba(X[te]); cls = list(e.classes_)
    return pr[:, cls.index(1)] if 1 in cls else np.zeros(len(te))

t0 = time.time(); out = {}
for pos, k in enumerate(eligible):
    yk = Y[:, k]
    skf = StratifiedKFold(n_splits=K, shuffle=True,
                          random_state=state(SEED, k, 0xF01D5))
    folds = [te for _, te in skf.split(X, yk)]
    for f in folds:                      # both classes in every training set
        tr = np.setdiff1d(np.arange(n), f)
        assert 0 < yk[tr].sum() < len(tr), f"degenerate fold k={k}"
    ests = pool_with_states([state(SEED, k, j) for j in range(20)])
    probs = np.zeros((n, 20), dtype=np.float32)
    jobs, meta = [], []
    for j, (name, est) in enumerate(ests):
        for te in folds:
            tr = np.setdiff1d(np.arange(n), te)
            jobs.append(delayed(fit_one)(est, tr, te, yk)); meta.append((j, te))
    res = Parallel(n_jobs=NJOBS, prefer="processes")(jobs)
    for (j, te), pr in zip(meta, res):
        probs[te, j] = pr
    out[f"probs_{k}"] = probs
    out[f"folds_{k}"] = np.array([np.pad(f, (0, n - len(f)),
                                  constant_values=-1) for f in folds])
    print(f"[{pos+1}/{len(eligible)}] k={k} P={P[k]} t={time.time()-t0:.0f}s",
          flush=True)

out["eligible"] = np.array(eligible); out["Y"] = Y.astype(np.int8)
fn = HERE / f"register_conf_seed{SEED}.npz"
np.savez_compressed(fn, **out)
print("DONE", f"{time.time()-t0:.0f}s sha256:",
      hashlib.sha256(fn.read_bytes()).hexdigest(), flush=True)
