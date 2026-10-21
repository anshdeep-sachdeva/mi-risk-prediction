"""Unsupervised learning: PCA + clustering to discover patient groups (risk phenotypes).

Labels are NOT used to build the clusters. We only look at outcomes afterwards, to ask:
  * Do natural patient groups differ in death / complication rates?
  * Does adding the cluster as an extra feature help the supervised model?
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from . import config as C
from .data import load_raw, make_target, select_features, split
from .models import make_preprocessor

N_PCS = 10          # clustering is done in a 10-D PCA space (removes noise from ~100 dims)
K_RANGE = range(2, 9)


class ClusterFeatures(BaseEstimator, TransformerMixin):
    """Fits PCA + K-Means on the (training) data and outputs one-hot cluster membership."""

    def __init__(self, k=3, n_pcs=N_PCS, random_state=C.RANDOM_STATE):
        self.k, self.n_pcs, self.random_state = k, n_pcs, random_state

    def fit(self, X, y=None):
        self.pca_ = PCA(self.n_pcs, random_state=self.random_state).fit(X)
        self.km_ = KMeans(self.k, n_init=20, random_state=self.random_state).fit(
            self.pca_.transform(X))
        return self

    def transform(self, X):
        lab = self.km_.predict(self.pca_.transform(X))
        return np.eye(self.k)[lab]


def run():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    feats = select_features(df, verbose=False)
    X = df[feats]
    died = make_target(df, C.PRIMARY_TARGET).values

    Z = make_preprocessor(feats).fit_transform(X)

    # ---- PCA ------------------------------------------------------------
    pca_full = PCA(random_state=C.RANDOM_STATE).fit(Z)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    pcs = pca_full.transform(Z)[:, :N_PCS]
    n90 = int(np.searchsorted(cum, 0.90) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(range(1, len(cum) + 1), 100 * cum, color="#2c6fbb")
    axes[0].axhline(90, color="#999", ls=":"); axes[0].axvline(n90, color="#999", ls=":")
    axes[0].set(xlabel="Number of principal components", ylabel="Cumulative variance (%)",
                title=f"PCA: {n90} components explain 90% of variance")
    order = np.argsort(died)  # draw deaths on top
    axes[1].scatter(pcs[order, 0], pcs[order, 1], c=np.where(died[order] == 1, "#c0392b", "#2c6fbb"),
                    s=7, alpha=0.55)
    axes[1].scatter([], [], c="#2c6fbb", s=12, label="Survived")
    axes[1].scatter([], [], c="#c0392b", s=12, label="Died")
    axes[1].legend(fontsize=8)
    axes[1].set(xlabel="PC 1", ylabel="PC 2", title="Patients in the first two PCs (labels not used)")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "09_pca.png"); plt.close(fig)

    # ---- choose k for K-Means -------------------------------------------
    sil, inertia = {}, {}
    for k in K_RANGE:
        km = KMeans(k, n_init=20, random_state=C.RANDOM_STATE).fit(pcs)
        sil[k] = silhouette_score(pcs, km.labels_, random_state=C.RANDOM_STATE)
        inertia[k] = km.inertia_
    # k = 2 trivially maximises silhouette on most data; we prefer k >= 3 for more
    # clinically useful groups, picking the best silhouette among k = 3..6.
    k_best = max(range(3, 7), key=lambda k: sil[k])

    fig, ax1 = plt.subplots(figsize=(6, 3.4))
    ax1.plot(list(K_RANGE), list(inertia.values()), "o-", color="#2c6fbb")
    ax1.set(xlabel="Number of clusters k", ylabel="Inertia (within-cluster SS)")
    ax2 = ax1.twinx()
    ax2.plot(list(K_RANGE), list(sil.values()), "s--", color="#c0392b")
    ax2.set_ylabel("Silhouette score", color="#c0392b")
    ax1.axvline(k_best, color="#999", ls=":")
    ax1.set_title(f"Choosing k (selected k = {k_best})")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "10_choose_k.png"); plt.close(fig)

    km = KMeans(k_best, n_init=20, random_state=C.RANDOM_STATE).fit(pcs)
    # Relabel clusters by death rate (cluster 1 = lowest risk) for readability.
    raw = km.labels_
    rank = pd.Series(died).groupby(raw).mean().sort_values().index
    remap = {old: new + 1 for new, old in enumerate(rank)}
    lab = np.array([remap[l] for l in raw])

    agg = AgglomerativeClustering(k_best, linkage="ward").fit_predict(pcs)
    ari = adjusted_rand_score(lab, agg)

    # ---- cluster profiles -----------------------------------------------
    prof = pd.DataFrame({"cluster": lab, "AGE": df.AGE, "died": died})
    for o in C.OUTCOMES:
        prof[o] = (df[o] > 0).astype(int)
    for c in ["K_SH_POST", "ZSN_A", "O_L_POST", "INF_ANAM", "SEX", "endocr_01", "ant_im",
              "S_AD_ORIT"]:
        prof[c] = df[c]
    g = prof.groupby("cluster")
    profile = pd.DataFrame({
        "patients": g.size(),
        "mean_age": g.AGE.mean().round(1),
        "male_pct": (100 * g.SEX.mean()).round(1),
        "prior_MI_pct": (100 * g.INF_ANAM.apply(lambda s: (s > 0).mean())).round(1),
        "prior_HF_pct": (100 * g.ZSN_A.apply(lambda s: (s > 0).mean())).round(1),
        "diabetes_pct": (100 * g.endocr_01.mean()).round(1),
        "shock_pct": (100 * g.K_SH_POST.mean()).round(1),
        "mean_ICU_systolic_BP": g.S_AD_ORIT.mean().round(0),
        "death_rate_pct": (100 * g.died.mean()).round(1),
        "heart_failure_pct": (100 * g.ZSN.mean()).round(1),
    })
    profile.to_csv(C.METRICS_DIR / "cluster_profiles.csv")
    chi2, pval, *_ = chi2_contingency(pd.crosstab(lab, died))

    # Heat-map of complication rates by cluster
    rates = 100 * g[C.OUTCOMES].mean().T
    rates.index = [C.OUTCOME_NAMES[o] for o in rates.index]
    fig, ax = plt.subplots(figsize=(1.3 * k_best + 3.2, 4.6))
    im = ax.imshow(rates.values, cmap="Reds", aspect="auto")
    ax.set_xticks(range(k_best)); ax.set_xticklabels([f"Cluster {c}\n(n={profile.patients[c]})"
                                                      for c in rates.columns], fontsize=8)
    ax.set_yticks(range(len(rates))); ax.set_yticklabels(rates.index, fontsize=8)
    for i in range(rates.shape[0]):
        for j in range(rates.shape[1]):
            v = rates.values[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=7,
                    color="white" if v > rates.values.max() * 0.6 else "black")
    ax.set_title("Complication rates per patient cluster", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, label="% of patients")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "11_cluster_outcomes.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4))
    cmap = plt.get_cmap("viridis", k_best)
    for c in range(1, k_best + 1):
        m = lab == c
        ax.scatter(pcs[m, 0], pcs[m, 1], s=7, alpha=0.6, color=cmap(c - 1),
                   label=f"Cluster {c} (death {profile.death_rate_pct[c]:.0f}%)")
    ax.set(xlabel="PC 1", ylabel="PC 2", title=f"K-Means clusters (k = {k_best}) in PCA space")
    ax.legend(fontsize=7, markerscale=2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "12_clusters_pca.png"); plt.close(fig)

    # ---- does cluster membership help the supervised model? ------------
    y = make_target(df, C.PRIMARY_TARGET)
    X_tr, X_te, y_tr, y_te = split(X, y)
    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)

    def lr():
        return LogisticRegression(C=0.03, class_weight="balanced", max_iter=5000)

    plain = Pipeline([("prep", make_preprocessor(feats)), ("clf", lr())])
    with_cl = Pipeline([
        ("prep", make_preprocessor(feats)),
        ("feats", FeatureUnion([("orig", FunctionTransformer()),
                                ("clusters", ClusterFeatures(k=k_best))])),
        ("clf", lr()),
    ])
    aug = {}
    for name, p in [("without_cluster_feature", plain), ("with_cluster_feature", with_cl)]:
        s = cross_val_score(p, X_tr, y_tr, cv=cv, scoring="roc_auc")
        test_auc = roc_auc_score(y_te, p.fit(X_tr, y_tr).predict_proba(X_te)[:, 1])
        aug[name] = {"cv_auc_mean": round(float(s.mean()), 4), "cv_auc_std": round(float(s.std()), 4),
                     "test_auc": round(float(test_auc), 4)}

    summary = {
        "pca_components_for_90pct_variance": n90,
        "pca_variance_first_2_pct": round(100 * float(cum[1]), 1),
        "pca_variance_first_10_pct": round(100 * float(cum[N_PCS - 1]), 1),
        "silhouette_by_k": {k: round(float(v), 3) for k, v in sil.items()},
        "k_selected": k_best,
        "kmeans_vs_agglomerative_ARI": round(float(ari), 3),
        "cluster_vs_death_chi2_p": float(pval),
        "cluster_profiles": json.loads(profile.to_json(orient="index")),
        "cluster_feature_experiment": aug,
    }
    with open(C.METRICS_DIR / "unsupervised_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  PCA: {n90} PCs for 90% variance | k = {k_best} | ARI vs Ward = {ari:.2f} | "
          f"cluster-vs-death chi2 p = {pval:.2e}")
    print(profile.to_string())
    print(f"  cluster feature experiment: {aug}")
    return summary
