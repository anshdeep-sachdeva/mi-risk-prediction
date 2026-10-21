"""Live demo: predict death risk for unseen (test-set) patients with the trained model.

Examples
    python demo.py                      # 5 random test patients
    python demo.py --n 10 --seed 7      # 10 different random patients
    python demo.py --high-risk          # show the patients the model flags as highest risk
    python demo.py --patient 1234       # a specific patient (row index from the dataset)
    python demo.py --patient 1234 --set AGE=80 K_SH_POST=1
                                        # "what-if": change values and see the risk move
Run `python main.py` first (it trains and saves models/mortality_model.joblib).
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from src import config as C

KEY_FIELDS = ["AGE", "SEX", "INF_ANAM", "ZSN_A", "K_SH_POST", "O_L_POST", "S_AD_ORIT",
              "ant_im", "TIME_B_S", "L_BLOOD"]


def band(p, thr):
    if p >= max(0.6, thr):
        return "HIGH RISK"
    if p >= thr:
        return "ELEVATED (flagged)"
    return "lower risk"


def show(idx, row, p, thr, actual=None):
    print("-" * 64)
    print(f"Patient #{idx}")
    for f in KEY_FIELDS:
        v = row.get(f, np.nan)
        v = "missing" if pd.isna(v) else (int(v) if float(v).is_integer() else round(float(v), 2))
        print(f"   {C.pretty(f):38s} {v}")
    print(f"   >> Predicted risk of death: {100 * p:5.1f}%   -> {band(p, thr)}")
    if actual is not None:
        print(f"   >> What actually happened:  {'DIED' if actual else 'survived'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patient", type=int)
    ap.add_argument("--high-risk", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="FEATURE=VALUE")
    a = ap.parse_args()

    path = C.MODEL_DIR / "mortality_model.joblib"
    if not path.exists():
        raise SystemExit("No trained model found. Run `python main.py` first.")
    bundle = joblib.load(path)
    model, feats, thr = bundle["model"], bundle["features"], bundle["threshold"]
    pts = pd.read_csv(C.MODEL_DIR / "demo_patients.csv", index_col=0)
    print(f"Model: {bundle['model_name']} | flags a patient when risk >= {100 * thr:.0f}% "
          f"(threshold chosen to catch ~80% of deaths)")

    if a.patient is not None:
        if a.patient not in pts.index:
            from src.data import load_raw, make_target
            df = load_raw()
            sel = df.loc[[a.patient], feats]
            actual = int(make_target(df, C.PRIMARY_TARGET).loc[a.patient])
            print("(note: this patient was in the TRAINING set)")
        else:
            sel = pts.loc[[a.patient], feats]
            actual = int(pts.loc[a.patient, "actual_died"])
        p0 = model.predict_proba(sel)[0, 1]
        show(a.patient, sel.iloc[0], p0, thr, actual)
        if a.set:
            mod = sel.copy()
            for kv in a.set:
                k, v = kv.split("=")
                mod[k] = float(v)
            p1 = model.predict_proba(mod)[0, 1]
            print(f"\nWHAT-IF with {' '.join(a.set)}:")
            show(a.patient, mod.iloc[0], p1, thr)
            print(f"   >> Risk changed by {100 * (p1 - p0):+.1f} percentage points")
        return

    probs = pd.Series(model.predict_proba(pts[feats])[:, 1], index=pts.index)
    if a.high_risk:
        chosen = probs.sort_values(ascending=False).index[: a.n]
    else:
        chosen = pts.sample(a.n, random_state=a.seed).index
    for i in chosen:
        show(i, pts.loc[i], probs[i], thr, int(pts.loc[i, "actual_died"]))
    flagged = probs >= thr
    caught = (flagged & (pts.actual_died == 1)).sum()
    print("=" * 64)
    print(f"Whole test set ({len(pts)} unseen patients): flagged {flagged.sum()}, "
          f"caught {caught} of {int(pts.actual_died.sum())} deaths.")


if __name__ == "__main__":
    main()
