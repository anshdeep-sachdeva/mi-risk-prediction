"""Unsupervised learning: PCA, K-Means, DBSCAN, hierarchical clustering.

We first run a paper-inspired baseline (raw, unscaled features: cholesterol and
heart rate dominate because of their large numeric range), then repeat it with scaling
(standardised numeric features + one-hot categories). Agreement between clusters and the
risk label is measured with the Adjusted Rand Index (ARI), as in the paper.
Component counts and DBSCAN epsilon are selected using those labels, so the
reported best ARIs are exploratory rather than independently validated scores.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from . import config as C
from .data import encoded, load

RS = C.RANDOM_STATE


def kmeans_ari_by_pcs(Z, y, max_pcs=10):
    out = {}
    for n in range(1, max_pcs + 1):
        P = PCA(n, random_state=RS).fit_transform(Z)
        lab = KMeans(2, n_init=20, random_state=RS).fit_predict(P)
        out[n] = adjusted_rand_score(y, lab)
    return out


def best_dbscan(Z, y, eps_grid, min_samples=5):
    best = (-1, None, None)
    for eps in eps_grid:
        lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)
        if len(set(lab)) < 2:
            continue
        a = adjusted_rand_score(y, lab)
        if a > best[0]:
            best = (a, eps, lab)
    return best


def run():
    df = load()
    y = df[C.TARGET].values
    X = df[C.FEATURES]
    Z_raw = encoded(X, scale=False).values          # paper-like: no scaling
    Z = encoded(X, scale=True).values               # proper: standardised + one-hot

    # ---- PCA ----------------------------------------------------------------
    pca_raw, pca = PCA(random_state=RS).fit(Z_raw), PCA(random_state=RS).fit(Z)
    P_raw, P = pca_raw.transform(Z_raw), pca.transform(Z)
    load_raw = pd.Series(np.abs(pca_raw.components_[0]), index=encoded(X, scale=False).columns)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, PP, pc, title in [(axes[0], P_raw, pca_raw, "Unscaled baseline"),
                              (axes[1], P, pca, "Standardised + one-hot (ours)")]:
        for lab, col, name in [(0, "#2c6fbb", "Low risk (0)"), (1, "#c0392b", "High risk (1)")]:
            ax.scatter(PP[y == lab, 0], PP[y == lab, 1], s=12, alpha=0.7, color=col, label=name)
        v = 100 * pc.explained_variance_ratio_[:2]
        ax.set(xlabel=f"PC1 ({v[0]:.0f}% var.)", ylabel=f"PC2 ({v[1]:.0f}% var.)", title=title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle("PCA of 303 patients, coloured by label (labels NOT used by PCA)", fontsize=11)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "09_pca_raw_vs_scaled.png"); plt.close(fig)

    # ---- K-Means (k = 2) vs number of PCs, raw vs scaled -------------------
    ari_raw = kmeans_ari_by_pcs(Z_raw, y)
    ari_sc = kmeans_ari_by_pcs(Z, y)
    km_full = KMeans(2, n_init=20, random_state=RS).fit_predict(Z)
    ari_full = adjusted_rand_score(y, km_full)

    # ---- DBSCAN --------------------------------------------------------------
    db_raw = best_dbscan(Z_raw, y, np.arange(1, 101, 1))
    db_sc = best_dbscan(Z, y, np.round(np.arange(0.5, 6.01, 0.1), 2))

    # ---- Hierarchical (Ward) -----------------------------------------------
    agg = AgglomerativeClustering(2, linkage="ward").fit_predict(Z)
    ari_agg = adjusted_rand_score(y, agg)

    # ---- silhouette for k ---------------------------------------------------
    sil = {k: silhouette_score(Z, KMeans(k, n_init=20, random_state=RS).fit_predict(Z)) for k in range(2, 7)}

    best_n = max(ari_sc, key=ari_sc.get)
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(list(ari_raw), list(ari_raw.values()), "o-", color="#999999", label="Unscaled (paper-like)")
    ax.plot(list(ari_sc), list(ari_sc.values()), "o-", color="#c0392b", label="Standardised + one-hot")
    ax.axhline(C.PAPER_ARI["K-Means"], color="#999999", ls=":", lw=1)
    ax.text(10, C.PAPER_ARI["K-Means"] + 0.01, "paper: 0.02", ha="right", fontsize=8, color="#666")
    ax.set(xlabel="Number of principal components used", ylabel="ARI (clusters vs label)",
           title="K-Means (k = 2): agreement with the risk label")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "10_kmeans_ari_vs_pcs.png"); plt.close(fig)

    # ---- profile the scaled K-Means clusters -------------------------------
    lab = KMeans(2, n_init=20, random_state=RS).fit_predict(PCA(best_n, random_state=RS).fit_transform(Z))
    if df.loc[lab == 0, C.TARGET].mean() > df.loc[lab == 1, C.TARGET].mean():
        lab = 1 - lab                                  # cluster "B" = higher share labelled high risk
    raw = pd.read_csv(C.DATA_PATH)
    prof = pd.DataFrame({
        "patients": pd.Series(lab).value_counts().sort_index().values,
        "high_risk_pct": [100 * y[lab == k].mean() for k in (0, 1)],
        "mean_age": [df.age[lab == k].mean() for k in (0, 1)],
        "male_pct": [100 * df.sex[lab == k].mean() for k in (0, 1)],
        "mean_max_hr": [df.thalachh[lab == k].mean() for k in (0, 1)],
        "exercise_angina_pct": [100 * df.exng[lab == k].mean() for k in (0, 1)],
        "mean_oldpeak": [df.oldpeak[lab == k].mean() for k in (0, 1)],
        "mean_ca": [df.ca[lab == k].mean() for k in (0, 1)],
        "reversible_defect_pct": [100 * (df.thall[lab == k] == 3).mean() for k in (0, 1)],
    }, index=["Cluster A", "Cluster B"]).round(1)
    prof.to_csv(C.METRICS_DIR / "cluster_profiles.csv")

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for k, col, name in [(0, "#2c6fbb", "A"), (1, "#c0392b", "B")]:
        ax.scatter(P[lab == k, 0], P[lab == k, 1], s=14, alpha=0.7, color=col,
                   label=f"Cluster {name}: {prof.high_risk_pct.iloc[k]:.0f}% high risk (n={prof.patients.iloc[k]})")
    ax.set(xlabel="PC1", ylabel="PC2", title="K-Means clusters (standardised data)")
    ax.legend(fontsize=7.5); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "11_kmeans_clusters.png"); plt.close(fig)

    # ---- ARI summary bar -----------------------------------------------------
    table = pd.DataFrame([
        ["K-Means", C.PAPER_ARI["K-Means"], max(ari_raw.values()), max(ari_sc.values())],
        ["DBSCAN", C.PAPER_ARI["DBSCAN"], db_raw[0], db_sc[0]],
        ["Hierarchical (Ward)", np.nan, np.nan, ari_agg],
    ], columns=["method", "paper", "ours_unscaled", "ours_scaled"]).round(3)
    table.to_csv(C.METRICS_DIR / "clustering_ari.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    xs = np.arange(len(table)); w = 0.27
    for j, (col, name, c) in enumerate([("paper", "Paper", "#bbbbbb"), ("ours_unscaled", "Ours, unscaled", "#7f8c8d"),
                                        ("ours_scaled", "Ours, standardised", "#c0392b")]):
        vals = table[col].fillna(0).values
        ax.bar(xs + (j - 1) * w, vals, w, color=c, label=name)
        for xi, v, orig in zip(xs, vals, table[col].values):
            ax.text(xi + (j - 1) * w, v + 0.008, "n/a" if pd.isna(orig) else f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(xs); ax.set_xticklabels(table.method); ax.set_ylabel("Best ARI vs label")
    ax.set_title("Clustering agreement before and after scaling", fontsize=10)
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "12_clustering_ari_summary.png"); plt.close(fig)

    summary = {
        "pca_unscaled_var_pc1_pc2": [round(float(v), 3) for v in pca_raw.explained_variance_ratio_[:2]],
        "pca_unscaled_pc1_top_loadings": load_raw.sort_values(ascending=False).head(3).round(3).to_dict(),
        "pca_scaled_var_pc1_pc2": [round(float(v), 3) for v in pca.explained_variance_ratio_[:2]],
        "kmeans_ari_by_pcs_unscaled": {k: round(v, 3) for k, v in ari_raw.items()},
        "kmeans_ari_by_pcs_scaled": {k: round(v, 3) for k, v in ari_sc.items()},
        "kmeans_ari_all_features_scaled": round(ari_full, 3), "best_n_pcs_scaled": best_n,
        "dbscan_unscaled": {"ari": round(db_raw[0], 3), "eps": float(db_raw[1])},
        "dbscan_scaled": {"ari": round(db_sc[0], 3), "eps": float(db_sc[1]),
                          "n_clusters": int(len(set(db_sc[2])) - (1 if -1 in db_sc[2] else 0)),
                          "noise_points": int((db_sc[2] == -1).sum())},
        "ward_ari_scaled": round(ari_agg, 3),
        "silhouette_by_k": {k: round(v, 3) for k, v in sil.items()},
        "cluster_profiles": json.loads(prof.to_json(orient="index")),
    }
    json.dump(summary, open(C.METRICS_DIR / "unsupervised_summary.json", "w"), indent=2)
    print(f"  K-Means best ARI: unscaled {max(ari_raw.values()):.3f} | scaled {max(ari_sc.values()):.3f} "
          f"(n_pcs={best_n}) | all-features {ari_full:.3f}")
    print(f"  DBSCAN best ARI: unscaled {db_raw[0]:.3f} (eps {db_raw[1]}) | scaled {db_sc[0]:.3f} (eps {db_sc[1]})")
    print(f"  Ward ARI {ari_agg:.3f} | silhouette {summary['silhouette_by_k']}")
    print(prof.to_string())
    return summary
