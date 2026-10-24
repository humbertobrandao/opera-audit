"""Heterogeneous pool m=20 (pre-protocol DRAFT; frozen in protocol v3).

20 fixed learning configurations, all with predict_proba, each trained
per label (binary relevance). random_state = seed + index.
"""
SEED = 20260811


def pool(seed: int):
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                                  HistGradientBoostingClassifier)
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.neighbors import KNeighborsClassifier

    cfg = []
    hidden = [(32,), (64,), (128,), (64, 32), (16, 16), (256,)]
    for i, h in enumerate(hidden):
        cfg.append((f"mlp_{i+1}", MLPClassifier(hidden_layer_sizes=h,
                    learning_rate_init=1e-3, alpha=1e-4, max_iter=300,
                    solver="adam", random_state=seed + i)))
    for i, C in enumerate([0.01, 0.1, 1.0, 10.0]):
        cfg.append((f"logreg_{i+1}", LogisticRegression(C=C, max_iter=2000,
                    solver="liblinear", random_state=seed + 10 + i)))
    for i, (ne, md) in enumerate([(100, None), (100, 10), (300, None), (300, 10)]):
        cfg.append((f"rf_{i+1}", RandomForestClassifier(n_estimators=ne,
                    max_depth=md, n_jobs=1, random_state=seed + 20 + i)))
    for i, md in enumerate([None, 10]):
        cfg.append((f"et_{i+1}", ExtraTreesClassifier(n_estimators=200,
                    max_depth=md, n_jobs=1, random_state=seed + 30 + i)))
    cfg.append(("hgb_1", HistGradientBoostingClassifier(max_iter=200,
                random_state=seed + 40)))
    cfg.append(("nb_1", BernoulliNB(alpha=1.0)))
    cfg.append(("nb_2", BernoulliNB(alpha=0.1)))
    cfg.append(("knn_1", KNeighborsClassifier(n_neighbors=5)))
    assert len(cfg) == 20
    return cfg
