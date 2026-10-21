"""Supervised learning: train, tune and compare classifiers for MI outcome risk.

Protocol (no information from the test set is used for any decision):
  1. Stratified 80/20 train/test split.
  2. For each model: grid search with 5-fold stratified CV on the training set (ROC-AUC).
  3. Pick the best model by mean CV ROC-AUC.
  4. Choose a "high-sensitivity" decision threshold from out-of-fold training predictions
     (catch >= 80% of deaths), because missing a high-risk patient is the costly error.
  5. Evaluate every tuned model once on the held-out test set.
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from . import config as C
from .data import get_dataset, split
from .models import PARAM_GRIDS, make_pipeline, model_zoo

TARGET_RECALL = 0.80
COLORS = ["#7f8c8d", "#8e44ad", "#16a085", "#2c6fbb", "#c0392b"]


def metrics_at(y_true, prob, thr):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
        "recall": recall_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_true, pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, pred),
        "accuracy": (tp + tn) / len(y_true),
    }


def threshold_for_recall(y_true, prob, target=TARGET_RECALL):
    """Highest threshold whose recall is still >= target (fewest false alarms)."""
    prec, rec, thr = precision_recall_curve(y_true, prob)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else 0.5


def run(target=C.PRIMARY_TARGET, tag="mortality", extra_features=None, verbose=True):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y, feats = get_dataset(target, verbose=verbose)
    X_tr, X_te, y_tr, y_te = split(X, y)
    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    print(f"  train: {len(y_tr)} patients ({y_tr.sum()} positive) | "
          f"test: {len(y_te)} patients ({y_te.sum()} positive)")

    # Baseline that always predicts the majority class: shows why accuracy is misleading.
    base_acc = 1 - y_te.mean()

    rows, tuned, test_probs = [], {}, {}
    for name, clf in model_zoo().items():
        pipe = make_pipeline(feats, clf)
        gs = GridSearchCV(pipe, PARAM_GRIDS[name], scoring="roc_auc", cv=cv, n_jobs=1)
        gs.fit(X_tr, y_tr)
        i = gs.best_index_
        fold_scores = [gs.cv_results_[f"split{k}_test_score"][i] for k in range(C.CV_FOLDS)]
        best = gs.best_estimator_
        tuned[name] = best
        prob = best.predict_proba(X_te)[:, 1]
        test_probs[name] = prob
        m = metrics_at(y_te, prob, 0.5)
        rows.append({
            "model": name, "cv_roc_auc_mean": np.mean(fold_scores),
            "cv_roc_auc_std": np.std(fold_scores),
            **{f"test_{k}": v for k, v in m.items()},
            "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
        })
        print(f"  {name:22s} CV AUC {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f} | "
              f"test AUC {m['roc_auc']:.3f}  recall {m['recall']:.2f}  "
              f"precision {m['precision']:.2f}")

    res = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
    best_name = res.iloc[0]["model"]
    best = tuned[best_name]
    print(f"  -> best model by cross-validation: {best_name}")

    # High-sensitivity threshold, chosen from out-of-fold predictions on TRAIN only.
    oof = cross_val_predict(clone(best), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    thr = threshold_for_recall(y_tr, oof)
    best_prob = test_probs[best_name]
    m_default = metrics_at(y_te, best_prob, 0.5)
    m_sens = metrics_at(y_te, best_prob, thr)
    print(f"  high-sensitivity threshold = {thr:.3f}: test recall {m_sens['recall']:.2f}, "
          f"precision {m_sens['precision']:.2f}, specificity {m_sens['specificity']:.2f}")

    # ---------------- plots ----------------
    _plot_model_comparison(res, tag)
    _plot_roc_pr(y_te, test_probs, res, tag)
    _plot_confusions(y_te, best_prob, thr, best_name, tag)
    importance = _plot_importance(best, X_te, y_te, best_name, tag)

    # Effect of class weighting (imbalance handling) - quick ablation for discussion.
    imbalance = _imbalance_ablation(feats, X_tr, y_tr, X_te, y_te)

    # ---------------- save ----------------
    res_out = res.copy()
    res_out["best_params"] = res_out["best_params"].astype(str)
    res_out.round(4).to_csv(C.METRICS_DIR / f"{tag}_model_comparison.csv", index=False)
    summary = {
        "target": target, "task": tag, "n_features": len(feats),
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "positive_rate_pct": round(100 * float(y.mean()), 1),
        "majority_baseline_accuracy": round(float(base_acc), 3),
        "best_model": best_name, "best_params": res.iloc[0]["best_params"],
        "threshold_default": 0.5, "threshold_high_sensitivity": round(thr, 4),
        "test_metrics_default": {k: round(float(v), 3) for k, v in m_default.items()},
        "test_metrics_high_sensitivity": {k: round(float(v), 3) for k, v in m_sens.items()},
        "top_features": importance,
        "imbalance_ablation": imbalance,
        "all_models": json.loads(res_out.round(4).to_json(orient="records")),
    }
    with open(C.METRICS_DIR / f"{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if tag == "mortality":
        joblib.dump({"model": best, "features": feats, "threshold": thr,
                     "model_name": best_name, "target": target},
                    C.MODEL_DIR / "mortality_model.joblib")
        X_te.assign(actual_died=y_te.values).to_csv(C.MODEL_DIR / "demo_patients.csv")
    return summary


# ---------------------------------------------------------------------------
def _plot_model_comparison(res, tag):
    r = res.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.barh(r["model"], r["cv_roc_auc_mean"], xerr=r["cv_roc_auc_std"], color="#2c6fbb",
            alpha=0.85, capsize=3, label="5-fold CV (train)")
    ax.scatter(r["test_roc_auc"], r["model"], color="#c0392b", zorder=3, label="Held-out test")
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("ROC-AUC")
    ax.set_title("Model comparison")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / f"{tag}_05_model_comparison.png"); plt.close(fig)


def _plot_roc_pr(y_te, probs, res, tag):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for (name, prob), col in zip(probs.items(), COLORS):
        fpr, tpr, _ = roc_curve(y_te, prob)
        axes[0].plot(fpr, tpr, color=col, lw=1.6, label=f"{name} ({roc_auc_score(y_te, prob):.3f})")
        p, r, _ = precision_recall_curve(y_te, prob)
        axes[1].plot(r, p, color=col, lw=1.6,
                     label=f"{name} ({average_precision_score(y_te, prob):.3f})")
    axes[0].plot([0, 1], [0, 1], "k:", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate (recall)",
                title="ROC curves (test set)")
    axes[1].axhline(y_te.mean(), color="k", ls=":", lw=1)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (test set)")
    for a in axes:
        a.legend(fontsize=7, loc="lower right" if a is axes[0] else "upper right")
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / f"{tag}_06_roc_pr_curves.png"); plt.close(fig)


def _plot_confusions(y_te, prob, thr, name, tag):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    labels = ["Survived", "Died"] if tag == "mortality" else ["No", "Yes"]
    for ax, t, title in [(axes[0], 0.5, "Default threshold (0.50)"),
                         (axes[1], thr, f"High-sensitivity threshold ({thr:.2f})")]:
        cm = confusion_matrix(y_te, (prob >= t).astype(int))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"{name}: test-set confusion matrices", fontsize=11)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / f"{tag}_07_confusion_matrices.png"); plt.close(fig)


def _plot_importance(model, X_te, y_te, name, tag, top=15):
    """Permutation importance: how much test ROC-AUC drops when a feature is shuffled."""
    pi = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=15,
                                random_state=C.RANDOM_STATE, n_jobs=1)
    imp = pd.DataFrame({"feature": X_te.columns, "mean": pi.importances_mean,
                        "std": pi.importances_std}).sort_values("mean", ascending=False)
    imp.round(5).to_csv(C.METRICS_DIR / f"{tag}_feature_importance.csv", index=False)
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.barh([C.pretty(f) for f in t["feature"]], t["mean"], xerr=t["std"], color="#c0392b",
            alpha=0.85, capsize=2)
    ax.set_xlabel("Drop in test ROC-AUC when feature is shuffled")
    ax.set_title(f"Most important admission features ({name})", fontsize=10)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / f"{tag}_08_feature_importance.png"); plt.close(fig)
    return [{"feature": r.feature, "name": C.pretty(r.feature), "importance": round(r.mean, 4)}
            for r in imp.head(top).itertuples()]


def _imbalance_ablation(feats, X_tr, y_tr, X_te, y_te):
    """Same models with and without class weighting: shows why imbalance handling matters."""
    out = {}
    for name, w_on in [("Logistic Regression", "balanced"), ("Random Forest", "balanced_subsample")]:
        for weighted in (False, True):
            if name == "Logistic Regression":
                clf = LogisticRegression(C=0.1, max_iter=5000,
                                         class_weight=w_on if weighted else None)
            else:
                clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                             random_state=C.RANDOM_STATE,
                                             class_weight=w_on if weighted else None)
            p = make_pipeline(feats, clf).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
            m = metrics_at(y_te, p, 0.5)
            out[f"{name} | {'weighted' if weighted else 'unweighted'}"] = {
                k: round(float(m[k]), 3) for k in ("roc_auc", "recall", "precision", "accuracy")}
    return out
