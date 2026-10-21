"""Exploratory data analysis for the heart dataset."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .data import encoded, load

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
RED, BLUE, GREY = "#c0392b", "#2c6fbb", "#999999"


def run():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    y = df[C.TARGET]
    out = {"patients": len(df), "features": len(C.FEATURES), "missing_values": int(df.isna().sum().sum()),
           "high_risk": int(y.sum()), "high_risk_pct": round(100 * y.mean(), 1),
           "female": int((df.sex == 0).sum()), "male": int((df.sex == 1).sum())}

    # 1. High-risk rate by categorical factors -------------------------------
    raw = pd.read_csv(C.DATA_PATH)
    panels = [("sex", "Sex"), ("cp", "Chest pain type"), ("exng", "Exercise-induced angina"),
              ("ca", "Major vessels (ca)"), ("thall", "Thallium test"), ("slope", "ST slope")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    rates = {}
    for ax, (col, title) in zip(axes.ravel(), panels):
        s = raw[col].map(C.THALL_NAMES) if col == "thall" else raw[col].astype(str)
        g = raw.groupby(s)[C.TARGET].agg(["mean", "size"]).sort_values("mean")
        rates[col] = {k: round(100 * v, 1) for k, v in g["mean"].items()}
        ax.barh([f"{i} (n={n})" for i, n in zip(g.index, g["size"])], 100 * g["mean"],
                color=[RED if m > y.mean() else BLUE for m in g["mean"]])
        ax.axvline(100 * y.mean(), color="k", ls=":", lw=1)
        ax.set_xlim(0, 100); ax.set_title(title, fontsize=10); ax.tick_params(labelsize=8)
        ax.set_xlabel("% labelled high risk", fontsize=8)
    fig.suptitle("Share of patients labelled high risk (dotted line = overall 54.5%)", fontsize=11)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "01_risk_by_category.png"); plt.close(fig)
    out["high_risk_pct_by_group"] = rates

    # 2. Numeric distributions by label --------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    for ax, col in zip(axes, ["age", "thalachh", "oldpeak", "chol"]):
        ax.hist(df.loc[y == 0, col], bins=20, alpha=0.6, color=BLUE, density=True, label="Low risk (0)")
        ax.hist(df.loc[y == 1, col], bins=20, alpha=0.6, color=RED, density=True, label="High risk (1)")
        ax.set_title(C.NAMES[col], fontsize=10); ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "02_numeric_by_label.png"); plt.close(fig)

    # 3. Correlation with the label ------------------------------------------
    Z = encoded(df[C.FEATURES], scale=False)
    corr = Z.apply(lambda c: np.corrcoef(c, y)[0, 1]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([C.pretty(c) for c in corr.index], corr.values,
            color=[RED if v > 0 else BLUE for v in corr.values])
    ax.axvline(0, color="k", lw=0.8); ax.tick_params(labelsize=8)
    ax.set_xlabel("Correlation with 'high risk' label")
    ax.set_title("Which features move with the label?", fontsize=11)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "03_correlation_with_label.png"); plt.close(fig)
    out["top_correlations"] = corr.round(3).iloc[np.argsort(-corr.abs().values)][:8].to_dict()
    return out
