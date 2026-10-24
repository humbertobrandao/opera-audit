"""Frozen pool m=20 (MLC protocol v3, Section 4).
pool_with_states(states): 20 estimators with an explicit random_state per
configuration (derived via SeedSequence by the caller). Everything not
listed uses the default of the pinned scikit-learn version.
"""

def pool_with_states(states):
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                                  HistGradientBoostingClassifier)
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.neighbors import KNeighborsClassifier
    assert len(states) == 20
    cfg, i = [], 0
    for h in [(32,), (64,), (128,), (64, 32), (16, 16), (256,)]:
        cfg.append((f"mlp_{i+1}", MLPClassifier(hidden_layer_sizes=h,
                    learning_rate_init=1e-3, alpha=1e-4, max_iter=300,
                    solver="adam", random_state=states[i]))); i += 1
    for C in [0.01, 0.1, 1.0, 10.0]:
        cfg.append((f"logreg_{i-5}", LogisticRegression(C=C, max_iter=2000,
                    solver="liblinear", random_state=states[i]))); i += 1
    for ne, md in [(100, None), (100, 10), (300, None), (300, 10)]:
        cfg.append((f"rf_{i-9}", RandomForestClassifier(n_estimators=ne,
                    max_depth=md, n_jobs=1, random_state=states[i]))); i += 1
    for md in [None, 10]:
        cfg.append((f"et_{i-13}", ExtraTreesClassifier(n_estimators=200,
                    max_depth=md, n_jobs=1, random_state=states[i]))); i += 1
    cfg.append(("hgb_1", HistGradientBoostingClassifier(max_iter=200,
                random_state=states[i]))); i += 1
    cfg.append(("nb_1", BernoulliNB(alpha=1.0))); i += 1
    cfg.append(("nb_2", BernoulliNB(alpha=0.1))); i += 1
    cfg.append(("knn_1", KNeighborsClassifier(n_neighbors=5))); i += 1
    assert len(cfg) == 20
    return cfg
