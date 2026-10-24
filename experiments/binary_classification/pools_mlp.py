"""
The pools of the single-label case study — Breast Cancer Wisconsin (Original).

PROTOCOL: THE WHOLE DATASET IS THE AUDITED HISTORY
--------------------------------------------------
Every instance is audited. There is no held-out partition, no development
subset and no instance withheld from the optimisation: the history is the
683 fully labelled records that remain after dropping those with a missing
attribute.

That raises the question of where the recorded predictions come from. Fitting
a member on the very records it then predicts would make the register
in-sample and inflate U* through memorisation. The register is therefore
produced the way a data scientist produces one when every label is known: by
stratified k-fold cross-validation, keeping the out-of-fold predictions. Each
record carries the verdict of a model that never saw it, and each record is
audited.

Note what cross-validation is and is not doing here. It builds the historical
register; it does not evaluate OPERA. No accuracy estimate from it is
reported, and the audit itself remains an exact optimisation over the
complete history, in the oracle regime of Section 3.

WHAT THE DESIGN HAS TO SUPPORT
------------------------------
A data scientist assembles a pool of five multilayer perceptrons, fits them
properly, and reads each one at the default probability threshold of 0.5 — as
if the two errors cost the same. The audit returns *no-go*. From there, each
natural attempt at a fix becomes another audit, and each isolates exactly one
hypothesis:

    pool 1  m = 5   under-parameterised networks, threshold 0.5
    pool 2  m = 5   under-parameterised networks, threshold chosen by cost
    pool 3  m = 5   well-fitted networks, threshold 0.5        <- paper: initial
    pool 4  m = 5   well-fitted networks, threshold by cost    <- paper: rebuilt
    pool 5  m = 25  under-parameterised networks, threshold 0.5
    pool 6  m = 25  well-fitted networks, threshold 0.5        <- paper: enlarged

The paper reports pools 3, 6 and 4, in that order. Pools 1, 2 and 5 belong to
the pre-registered grid and are kept because dropping them after seeing the
results would be selection.

From the initial pool to the rebuilt one the size does not grow: five
classifiers at the start, five at the end. What changes is the coherence
between how each classifier is read and the asymmetry of the costs.

OPERATING POINT (pools 2 and 4): CLOSED FORM, NOT TUNED
-------------------------------------------------------
Reading a probabilistic classifier at 0.5 is not a neutral default: it is the
optimal rule under one specific assumption, namely that the four outcomes are
worth the same. Once the four costs are declared, the optimal rule follows
from them in closed form — the cost-sensitive threshold of Elkan (2001):

    tau* = (r_tn - r_fp) / ((r_tp - r_fn) + (r_tn - r_fp))

Flagging record i is preferable to staying silent exactly when
p_i (r_tp - r_fn) >= (1 - p_i)(r_tn - r_fp), which rearranges to p_i >= tau*.

Nothing here is estimated, so nothing can leak: tau* is a function of the four
declared parameters alone. It touches neither the audited history nor any
validation data. With the costs of this study, tau* = 61 / 26161 ~ 0.0023,
against the 0.5 that the same classifiers were being read at.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

COLS = ["id", "thickness", "size_unif", "shape_unif", "adhesion", "epithelial",
        "bare_nuclei", "chromatin", "nucleoli", "mitoses", "label"]

SEED = 20260811
OUTER_FOLDS = 10          # register of predictions over the whole history


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, header=None, names=COLS, na_values="?")
    df = df.dropna().reset_index(drop=True)
    X = df[COLS[1:-1]].to_numpy(dtype=float)
    y = (df["label"].to_numpy() == 4).astype(int)     # 4 = malignant = positive
    return X, y


# ---------------------------------------------------------------------------
# Pool definitions. What makes a configuration "under-parameterised" is stated
# explicitly: eta between 1.5 and 10 (against a typical 1e-3) and 2 to 8 epochs
# (against thousands). No configuration was chosen after looking at an audit
# result; the robustness of that claim is what seeds_sweep.py measures.
# ---------------------------------------------------------------------------
def mlps_underfit(seed: int):
    from sklearn.neural_network import MLPClassifier
    cfg = [("mlp_bad_1", (2,),   3.0,  5),
           ("mlp_bad_2", (3,),   5.0,  3),
           ("mlp_bad_3", (2, 2), 2.0,  8),
           ("mlp_bad_4", (4,),  10.0,  2),
           ("mlp_bad_5", (2,),   1.5,  4)]
    return [(nm, MLPClassifier(hidden_layer_sizes=h, learning_rate_init=eta,
                               max_iter=it, solver="sgd", random_state=seed + i))
            for i, (nm, h, eta, it) in enumerate(cfg)]


def mlps_underfit_large(seed: int):
    from sklearn.neural_network import MLPClassifier
    out, i = list(mlps_underfit(seed)), 100
    for h in [(2,), (3,), (4,), (2, 2), (3, 2)]:
        for eta in [1.5, 2.0, 3.0, 5.0]:
            out.append((f"mlp_badx{h}_{eta}",
                        MLPClassifier(hidden_layer_sizes=h, learning_rate_init=eta,
                                      max_iter=2 + (i % 7), solver="sgd",
                                      random_state=seed + i)))
            i += 1
    return out[:25]


def mlps_good(seed: int):
    """The paper's initial pool: five properly fitted networks."""
    from sklearn.neural_network import MLPClassifier
    cfg = [("mlp_1", (32,),    1e-3, 2000, 1e-4),
           ("mlp_2", (64, 32), 1e-3, 2000, 1e-4),
           ("mlp_3", (16, 16), 5e-4, 3000, 1e-4),
           ("mlp_4", (128,),   1e-3, 2000, 1e-3),
           ("mlp_5", (64,),    1e-2, 1500, 1e-4)]
    return [(nm, MLPClassifier(hidden_layer_sizes=h, learning_rate_init=eta,
                               max_iter=it, alpha=a, solver="adam",
                               random_state=seed + i))
            for i, (nm, h, eta, it, a) in enumerate(cfg)]


def mlps_good_large(seed: int):
    """The paper's enlarged pool: the five networks of `mlps_good` plus twenty
    more, same reading.

    The enlarged pool is a STRICT SUPERSET of the initial one, and that is
    deliberate. Every decision feasible on five members stays feasible on
    twenty-five, so U*(enlarged) >= U*(initial) holds by construction. The
    comparison therefore cannot be confounded by having swapped classifiers:
    whatever the larger pool achieves, it achieves in addition to what the
    smaller one already could. It also gives a sanity check on the solver — a
    strictly smaller optimum on the superset would be a bug.
    """
    from sklearn.neural_network import MLPClassifier
    out, i = list(mlps_good(seed)), 100
    for h in [(16,), (32,), (64,), (64, 32), (16, 16)]:
        for eta, a in [(1e-3, 1e-4), (5e-4, 1e-4), (1e-2, 1e-4), (1e-3, 1e-3)]:
            out.append((f"mlp_x{h}_{eta}_{a}",
                        MLPClassifier(hidden_layer_sizes=h, learning_rate_init=eta,
                                      alpha=a, max_iter=2000, solver="adam",
                                      random_state=seed + i)))
            i += 1
    return out[:25]


# ---------------------------------------------------------------------------
def _fit(clf, X, y):
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.exceptions import ConvergenceWarning
    model = make_pipeline(StandardScaler(), clf)
    with warnings.catch_warnings():
        # non-convergence of the under-parameterised networks is the point
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y)
    return model


def _proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return 1.0 / (1.0 + np.exp(-model.decision_function(X)))


def bayes_tau(costs) -> float:
    """Cost-sensitive threshold implied by the four declared costs.

    Flagging beats staying silent iff  p (r_tp - r_fn) >= (1-p)(r_tn - r_fp),
    i.e. iff  p >= (r_tn - r_fp) / ((r_tp - r_fn) + (r_tn - r_fp)).

    At symmetric costs this returns 0.5, which is why 0.5 is the conventional
    default — and why it is the wrong reading as soon as the costs are not
    symmetric. No data is involved.
    """
    r_tp, r_tn, r_fn, r_fp = costs
    gain_flag = r_tp - r_fn          # value of flagging a positive
    gain_quiet = r_tn - r_fp         # value of staying silent on a negative
    return float(gain_quiet / (gain_flag + gain_quiet))


def record(pool, X, y, costs=None, seed: int = SEED):
    """Build the historical register of predictions over the WHOLE history.

    For each outer fold, every member is fitted on the remaining folds and
    predicts the held-out records. With `costs`, each member is read at the
    cost-sensitive threshold implied by those costs; without, at the default
    0.5. Neither reading is estimated from data, so no record influences its
    own prediction in any way.

    Returns the n x m binary matrix M, the member names, and the threshold.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    n, m = len(y), len(pool)
    M = np.zeros((n, m), dtype=int)
    tau = 0.5 if costs is None else bayes_tau(costs)
    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=seed)

    for tr, te in outer.split(X, y):
        for j, (_, clf) in enumerate(pool):
            M[te, j] = (_proba(_fit(clone(clf), X[tr], y[tr]), X[te]) >= tau).astype(int)

    return M, [nm for nm, _ in pool], [tau] * m


def cost_structure(npos: int, nneg: int, r_tp: float = 100.0, r_tn: float = 1.0):
    """Costs satisfying the two conditions the design requires.

      (D) dominance:   |r_fn| > r_tp|I+| + r_tn|I-|
      (T) non-trivial: r_tp|I+| + r_fp|I-| <= 0

    (D) belongs to the model — it expresses "a missed malignancy is not
    compensable" without an infinite parameter. (T) belongs to the experiment:
    without it a *go* verdict would be reachable by flagging the entire
    history, and nothing would have been demonstrated.
    """
    dom = r_tp * npos + r_tn * nneg
    triv = r_tp * npos / nneg
    r_fp = -float(np.ceil(triv / 10) * 10)
    r_fn = -float(np.ceil(dom / 1000) * 1000 + 1000)
    assert abs(r_fn) > dom and r_tp * npos + r_fp * nneg <= 0
    return np.array([r_tp, r_tn, r_fn, r_fp]), dom, triv


# (label, builder, threshold chosen by cost). Fixed before any audit was run.
POOLS = [
    ("1. five under-parameterised networks, threshold 0.5",  mlps_underfit,       False),
    ("2. five under-parameterised networks, cost-sensitive threshold", mlps_underfit,       True),
    ("3. five well-fitted networks, threshold 0.5",          mlps_good,           False),
    ("4. five well-fitted networks, cost-sensitive threshold",         mlps_good,           True),
    ("5. twenty-five under-parameterised networks, thr 0.5", mlps_underfit_large, False),
    ("6. twenty-five well-fitted networks, threshold 0.5",   mlps_good_large,     False),
]
