"""Supervised learning: tune, compare and evaluate nine model families.

Improvements over the reference paper's protocol (60/20/20 split, selection by
validation F1 on ~60 patients):
  * stratified 80/20 split; all tuning by REPEATED stratified 5-fold CV (15 folds) on the
    training set, so model choice is based on 15 estimates instead of one small validation set
  * every preprocessing step inside a Pipeline (no leakage from scaling)
  * test set used once; 95% bootstrap confidence interval reported for test ROC-AUC,
    because 61 test patients make single numbers noisy
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from . import config as C
from .data import get_xy, split
from .models import model_zoo, pipe

PALETTE = ["#1b2430", "#7f8c8d", "#c0392b", "#e67e22", "#2c6fbb", "#16a085", "#8e44ad", "#d4ac0d", "#5d6d7e"]


def metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return {"accuracy": accuracy_score(y, pred), "roc_auc": roc_auc_score(y, prob),
            "pr_auc": average_precision_score(y, prob), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred, zero_division=0)}


def bootstrap_auc(y, prob, n=2000, seed=C.RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y, prob = np.asarray(y), np.asarray(prob)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], prob[i]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def run():
    for d in (C.FIG_DIR, C.METRICS_DIR, C.MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    X, y = get_xy()
    X_tr, X_te, y_tr, y_te = split(X, y)
    cv = RepeatedStratifiedKFold(n_splits=C.CV_SPLITS, n_repeats=C.CV_REPEATS, random_state=C.RANDOM_STATE)
    print(f"  train {len(y_tr)} ({y_tr.sum()} high-risk) | test {len(y_te)} ({y_te.sum()} high-risk)")

    rows, fitted, probs = [], {}, {}
    for name, (clf, grid) in model_zoo().items():
        gs = GridSearchCV(pipe(clf), grid, cv=cv, scoring={"roc_auc": "roc_auc", "f1": "f1"},
                          refit="roc_auc", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        i = gs.best_index_
        r = gs.cv_results_
        prob = gs.best_estimator_.predict_proba(X_te)[:, 1]
        fitted[name], probs[name] = gs.best_estimator_, prob
        m = metrics(y_te, prob)
        rows.append({"model": name,
                     "cv_roc_auc": r["mean_test_roc_auc"][i], "cv_roc_auc_std": r["std_test_roc_auc"][i],
                     "cv_f1": r["mean_test_f1"][i],
                     **{f"test_{k}": v for k, v in m.items()},
                     "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}})
        print(f"  {name:24s} CV AUC {r['mean_test_roc_auc'][i]:.3f}±{r['std_test_roc_auc'][i]:.3f} "
              f"F1 {r['mean_test_f1'][i]:.3f} | test AUC {m['roc_auc']:.3f} acc {m['accuracy']:.3f} "
              f"rec {m['recall']:.2f} prec {m['precision']:.2f} F1 {m['f1']:.3f}")

    res = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)
    best = res.loc[0, "model"]
    ci = bootstrap_auc(y_te, probs[best])
    print(f"  -> best by CV ROC-AUC: {best} (test AUC 95% CI {ci})")

    _plot_comparison(res)
    _plot_curves(y_te, probs, res)
    _plot_confusion(y_te, probs[best], best)
    odds = _plot_logreg(fitted["Logistic Regression"])
    imp = _plot_permutation(fitted[best], X_te, y_te, best)
    paper = _paper_table(res)

    out = res.copy(); out["best_params"] = out["best_params"].astype(str)
    out.round(4).to_csv(C.METRICS_DIR / "model_comparison.csv", index=False)
    paper.to_csv(C.METRICS_DIR / "comparison_with_paper.csv", index=False)
    summary = {"n_train": int(len(y_tr)), "n_test": int(len(y_te)), "best_model": best,
               "best_test_auc_95ci": ci,
               "best_test_metrics": {k: round(float(v), 3) for k, v in metrics(y_te, probs[best]).items()},
               "logreg_test_metrics": {k: round(float(v), 3) for k, v in metrics(y_te, probs["Logistic Regression"]).items()},
               "logreg_test_auc_95ci": bootstrap_auc(y_te, probs["Logistic Regression"]),
               "logreg_odds_ratios": odds, "permutation_importance": imp,
               "models": json.loads(out.round(4).to_json(orient="records"))}
    json.dump(summary, open(C.METRICS_DIR / "supervised_summary.json", "w"), indent=2, default=str)
    joblib.dump({"model": fitted[best], "name": best, "features": C.FEATURES},
                C.MODEL_DIR / "heart_model.joblib")
    X_te.assign(actual=y_te.values).to_csv(C.MODEL_DIR / "demo_patients.csv")
    return summary


def _plot_comparison(res):
    r = res.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(r["model"], r["cv_roc_auc"], xerr=r["cv_roc_auc_std"], color="#2c6fbb", alpha=0.85,
            capsize=3, label="Repeated 5-fold CV (train)")
    ax.scatter(r["test_roc_auc"], r["model"], color="#c0392b", zorder=3, label="Held-out test")
    ax.set_xlim(0.6, 1.0); ax.set_xlabel("ROC-AUC")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2, frameon=False)
    ax.set_title("Model comparison (9 model families)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "04_model_comparison.png"); plt.close(fig)


def _plot_curves(y, probs, res):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for (name, col) in zip(res["model"], PALETTE):
        p = probs[name]
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=col, lw=1.4, label=f"{name} ({roc_auc_score(y, p):.2f})")
        pr, rc, _ = precision_recall_curve(y, p)
        axes[1].plot(rc, pr, color=col, lw=1.4, label=f"{name} ({average_precision_score(y, p):.2f})")
    axes[0].plot([0, 1], [0, 1], "k:", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test set)")
    axes[1].axhline(y.mean(), color="k", ls=":", lw=1)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (test set)")
    for a in axes:
        a.legend(fontsize=6.5, loc="lower right" if a is axes[0] else "lower left")
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "05_roc_pr_curves.png"); plt.close(fig)


def _plot_confusion(y, prob, name):
    fig, ax = plt.subplots(figsize=(4, 3.6))
    cm = confusion_matrix(y, (prob >= 0.5).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["Low risk", "High risk"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}: test set", fontsize=10)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "06_confusion_matrix.png"); plt.close(fig)


def _plot_logreg(model):
    names = model.named_steps["prep"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_[0]
    s = pd.Series(np.exp(coef), index=names).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([C.pretty(n) for n in s.index], s.values, color=["#2c6fbb" if v < 1 else "#c0392b" for v in s.values])
    ax.axvline(1, color="k", lw=0.8); ax.set_xscale("log"); ax.tick_params(labelsize=8)
    ticks = [t for t in (0.25, 0.5, 1, 2, 4) if s.min() / 1.2 <= t <= s.max() * 1.2]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}" for t in ticks]); ax.minorticks_off()
    ax.set_xlabel("Odds ratio for 'high risk' (per 1 SD for numeric features, log scale)")
    ax.set_title("Logistic regression: what raises or lowers the odds", fontsize=10)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "07_logreg_odds_ratios.png"); plt.close(fig)
    return {C.pretty(k): round(float(v), 3) for k, v in s.items()}


def _plot_permutation(model, X, y, name):
    pi = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=30, random_state=C.RANDOM_STATE)
    s = pd.DataFrame({"f": X.columns, "m": pi.importances_mean, "s": pi.importances_std}).sort_values("m")
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.barh([C.NAMES[f] for f in s["f"]], s["m"], xerr=s["s"], color="#c0392b", alpha=0.85, capsize=2)
    ax.set_xlabel("Drop in test ROC-AUC when shuffled"); ax.tick_params(labelsize=8)
    ax.set_title(f"Permutation importance ({name})", fontsize=10)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "08_permutation_importance.png"); plt.close(fig)
    return {C.NAMES[f]: round(float(m), 4) for f, m in zip(s["f"][::-1], s["m"][::-1])}


def _paper_table(res):
    rows = []
    for name, p in C.PAPER_TEST.items():
        o = res.set_index("model").loc[name]
        rows.append({"model": name,
                     **{f"paper_{k}": v for k, v in p.items()},
                     **{f"ours_{k}": round(float(o[f"test_{k}"]), 3) for k in p}})
    return pd.DataFrame(rows)
