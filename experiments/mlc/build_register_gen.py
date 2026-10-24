"""Multi-dataset OOF register (generality+scale protocol, sha256 304b2e52).

Derivation: SeedSequence([seed, label, config]) and folds
SeedSequence([seed, label, 0xF01D5]). Preprocessing by the frozen rule:
continuous dataset -> Pipeline(StandardScaler fitted on the training folds
only, estimator), a single pipeline for the 20 configurations; binary
dataset -> as is. Measures T_wall and T_fit_sum (S-8). Hard training
failure -> TECHNICAL FAILURE of the seed (.FAILED file with the error),
with no replacement.

Usage: build_register_gen.py BASE SEED [NJOBS]
"""
import sys, time, json, hashlib, traceback
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
from pool_frozen import pool_with_states

BASE = sys.argv[1]
SEED = int(sys.argv[2])
NJOBS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
K = 10
HERE = Path(__file__).parent
OUT = HERE / "registers_gen"; OUT.mkdir(exist_ok=True)

censo = json.loads((HERE / "data_gen" / "censo_elegibilidade.json").read_text())[BASE]
CONTINUA = censo["tipo_atributo"] == "continuo"
d = np.load(HERE / "data_gen" / f"{BASE}_undivided.npz", allow_pickle=True)
X = d["X"].astype(np.float32)
Y = d["Y"].astype(int)
n, c = Y.shape
eligible = censo["elegiveis"]
print(f"{BASE} seed={SEED} n={n} p={X.shape[1]} L_eleg={len(eligible)} "
      f"type={'continuous' if CONTINUA else 'binary'} njobs={NJOBS}", flush=True)


def state(*args):
    return int(np.random.SeedSequence(list(args)).generate_state(1)[0] % (2**31))


def fit_one(est, tr, te, yk):
    """Returns (test_probs, fit_seconds). Runs in a worker."""
    from sklearn.base import clone
    e = clone(est)
    if CONTINUA:
        e = Pipeline([("scaler", StandardScaler()), ("est", e)])
    t0 = time.perf_counter()
    e.fit(X[tr], yk[tr])
    dt = time.perf_counter() - t0
    pr = e.predict_proba(X[te])
    cls = list(e[-1].classes_ if CONTINUA else e.classes_)
    p1 = pr[:, cls.index(1)] if 1 in cls else np.zeros(len(te))
    return p1, dt


t_wall0 = time.time()
out, fit_sum = {}, 0.0
try:
    for pos, k in enumerate(eligible):
        yk = Y[:, k]
        skf = StratifiedKFold(n_splits=K, shuffle=True,
                              random_state=state(SEED, k, 0xF01D5))
        folds = [te for _, te in skf.split(X, yk)]
        for f in folds:
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
        for (j, te), (pr, dt) in zip(meta, res):
            probs[te, j] = pr
            fit_sum += dt
        out[f"probs_{k}"] = probs
        out[f"folds_{k}"] = np.array([np.pad(f, (0, n - len(f)),
                                      constant_values=-1) for f in folds])
        print(f"[{pos+1}/{len(eligible)}] k={k} P={int(yk.sum())} "
              f"t_wall={time.time()-t_wall0:.0f}s", flush=True)
except Exception:
    err = traceback.format_exc()
    (OUT / f"register_{BASE}_seed{SEED}.FAILED").write_text(err)
    print("TECHNICAL FAILURE:\n" + err, flush=True)
    sys.exit(2)

out["eligible"] = np.array(eligible)
out["Y"] = Y.astype(np.int8)
fn = OUT / f"register_{BASE}_seed{SEED}.npz"
np.savez_compressed(fn, **out)
t_wall = time.time() - t_wall0
timing = {"base": BASE, "seed": SEED, "njobs": NJOBS,
          "T_wall_s": round(t_wall, 1), "T_fit_sum_s": round(fit_sum, 1),
          "L_eleg": len(eligible),
          "sha256": hashlib.sha256(fn.read_bytes()).hexdigest()}
(OUT / f"timing_{BASE}_seed{SEED}.json").write_text(json.dumps(timing, indent=1))
print(f"DONE T_wall={t_wall:.0f}s T_fit_sum={fit_sum:.0f}s "
      f"sha256={timing['sha256'][:16]}...", flush=True)
