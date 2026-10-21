"""How predictable is EACH of the 12 complications from admission data?

A quick screening: 5-fold stratified CV ROC-AUC for two models on every outcome.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from . import config as C
from .data import load_raw, make_target, select_features
from .models import make_pipeline


def run():
    df = load_raw()
    feats = select_features(df, verbose=False)
    X = df[feats]
    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    rows = []
    for o in C.OUTCOMES:
        y = make_target(df, o)
        r = {"outcome": C.OUTCOME_NAMES[o], "code": o, "positives": int(y.sum())}
        for name, clf in [
            ("Logistic Regression", LogisticRegression(C=0.03, class_weight="balanced", max_iter=5000)),
            ("Gradient Boosting", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=200, max_leaf_nodes=7, class_weight="balanced",
                random_state=C.RANDOM_STATE)),
        ]:
            s = cross_val_score(make_pipeline(feats, clf), X, y, cv=cv, scoring="roc_auc")
            r[name] = round(float(s.mean()), 3)
        rows.append(r)
        print(f"  {r['outcome']:30s} n+={r['positives']:4d}  LR {r['Logistic Regression']:.3f}  "
              f"GB {r['Gradient Boosting']:.3f}")
    res = pd.DataFrame(rows).sort_values("Gradient Boosting")
    res.to_csv(C.METRICS_DIR / "all_outcomes_auc.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    yy = np.arange(len(res))
    ax.barh(yy - 0.2, res["Logistic Regression"], 0.4, color="#7f8c8d", label="Logistic Regression")
    ax.barh(yy + 0.2, res["Gradient Boosting"], 0.4, color="#c0392b", label="Gradient Boosting")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{o} (n={p})" for o, p in zip(res.outcome, res.positives)], fontsize=8)
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.set_xlim(0.4, 1.0); ax.set_xlabel("5-fold CV ROC-AUC")
    ax.set_title("Predictability of every complication from admission data", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "13_all_outcomes_auc.png"); plt.close(fig)
    with open(C.METRICS_DIR / "all_outcomes_summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows
