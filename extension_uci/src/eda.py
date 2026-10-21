"""Exploratory data analysis: class balance, missingness, key risk-factor plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as C
from .data import load_raw, make_target

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
RED, BLUE, GREY = "#c0392b", "#2c6fbb", "#999999"


def run():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    summary = {"patients": len(df), "input_features": len(C.INPUTS),
               "overall_missing_pct": round(100 * df[C.INPUTS].isna().mean().mean(), 1)}

    # 1. Outcome prevalence (all 12 complications) -------------------------
    prev = pd.Series({C.OUTCOME_NAMES[o]: 100 * (df[o] > 0).mean() for o in C.OUTCOMES})
    prev = prev.sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = [RED if n.startswith("Death") else BLUE for n in prev.index]
    ax.barh(prev.index, prev.values, color=colors)
    for i, v in enumerate(prev.values):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("% of patients")
    ax.set_title("How common is each complication? (n = 1,700)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "01_outcome_prevalence.png"); plt.close(fig)

    # 2. Missingness -------------------------------------------------------
    miss = (100 * df[C.INPUTS].isna().mean()).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh([C.pretty(c) for c in miss.index][::-1], miss.values[::-1], color=GREY)
    ax.axvline(90, color=RED, ls="--", lw=1)
    ax.text(89, 0, "dropped (>90%)", color=RED, ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("% missing")
    ax.set_title("20 features with the most missing values")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "02_missingness.png"); plt.close(fig)

    # 3. Death rate by key admission findings ------------------------------
    died = make_target(df, C.PRIMARY_TARGET)
    groups = {
        "Age < 60": df.AGE < 60, "Age 60-70": df.AGE.between(60, 70),
        "Age > 70": df.AGE > 70,
        "No shock": df.K_SH_POST == 0, "Cardiogenic shock": df.K_SH_POST == 1,
        "No prior HF": df.ZSN_A == 0, "Prior HF": df.ZSN_A > 0,
        "No pulm. edema": df.O_L_POST == 0, "Pulmonary edema": df.O_L_POST == 1,
    }
    rates = pd.Series({k: 100 * died[m].mean() for k, m in groups.items()})
    ns = pd.Series({k: int(m.sum()) for k, m in groups.items()})
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    pal = [BLUE, BLUE, BLUE, GREY, RED, GREY, RED, GREY, RED]
    ax.bar(range(len(rates)), rates.values, color=pal)
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels([f"{k}\n(n={ns[k]})" for k in rates.index], fontsize=7, rotation=30, ha="right")
    for i, v in enumerate(rates.values):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=7)
    ax.axhline(100 * died.mean(), color="k", ls=":", lw=1)
    ax.text(len(rates) - 0.5, 100 * died.mean() + 1, f"overall {100*died.mean():.1f}%",
            ha="right", fontsize=8)
    ax.set_ylabel("Death rate (%)")
    ax.set_title("Death rate rises sharply with age, shock and heart failure")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "03_death_rate_by_factor.png"); plt.close(fig)

    # 4. Age distribution by outcome --------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(df.AGE[died == 0].dropna(), bins=25, alpha=0.6, color=BLUE, label="Survived",
            density=True)
    ax.hist(df.AGE[died == 1].dropna(), bins=25, alpha=0.6, color=RED, label="Died",
            density=True)
    ax.set_xlabel("Age (years)"); ax.set_ylabel("Density"); ax.legend()
    ax.set_title("Patients who died were older on average")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "04_age_by_outcome.png"); plt.close(fig)

    summary.update({
        "deaths": int(died.sum()), "death_rate_pct": round(100 * died.mean(), 1),
        "death_rate_by_group_pct": rates.round(1).to_dict(),
    })
    return summary
