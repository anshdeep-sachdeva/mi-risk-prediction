"""Live demo: predict MI-risk label for unseen (test-set) patients with the trained model.

Examples
    python demo.py                         # 5 random test patients
    python demo.py --n 8 --seed 3
    python demo.py --patient 10            # one patient (row index in the dataset)
    python demo.py --patient 10 --set thalachh=120 exng=Yes cp=Non-Angina
                                           # what-if: change values and see the prediction move
Run `python main.py` first (it trains and saves models/heart_model.joblib).
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

import joblib
import pandas as pd

from src import config as C
from src.data import load

SHOW = ["age", "sex", "cp", "trtbps", "chol", "thalachh", "exng", "oldpeak", "ca", "thall"]


def readable(row):
    out = {}
    for f in SHOW:
        v = row[f]
        if f == "sex":
            v = "Male" if v == 1 else "Female"
        elif f == "exng":
            v = "Yes" if v == 1 else "No"
        elif f == "thall":
            v = C.THALL_NAMES.get(int(v), v)
        out[C.NAMES[f]] = v
    return out


def show(idx, row, p, actual=None):
    print("-" * 60)
    print(f"Patient #{idx}")
    for k, v in readable(row).items():
        print(f"   {k:30s} {v}")
    label = "HIGH RISK (1)" if p >= 0.5 else "low risk (0)"
    print(f"   >> Predicted probability of 'high risk': {100 * p:5.1f}%  -> {label}")
    if actual is not None:
        ok = (p >= 0.5) == bool(actual)
        print(f"   >> True label: {'HIGH RISK (1)' if actual else 'low risk (0)'}   "
              f"[{'correct' if ok else 'WRONG'}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patient", type=int)
    ap.add_argument("--set", nargs="*", default=[], metavar="FEATURE=VALUE")
    a = ap.parse_args()
    path = C.MODEL_DIR / "heart_model.joblib"
    if not path.exists():
        raise SystemExit("No trained model found. Run `python main.py` first.")
    b = joblib.load(path)
    model = b["model"]
    pts = pd.read_csv(C.MODEL_DIR / "demo_patients.csv", index_col=0)
    print(f"Model: {b['name']} (tuned with repeated 5-fold cross-validation)")
    print("Labels follow the reference dataset; probabilities are not validated clinical MI risk.")

    if a.patient is not None:
        if a.patient in pts.index:
            row, actual = pts.loc[[a.patient], C.FEATURES], int(pts.loc[a.patient, "actual"])
        else:
            df = load()
            row, actual = df.loc[[a.patient], C.FEATURES], int(df.loc[a.patient, C.TARGET])
            print("(note: this patient was in the TRAINING set)")
        p0 = model.predict_proba(row)[0, 1]
        show(a.patient, row.iloc[0], p0, actual)
        if a.set:
            mod = row.copy()
            for kv in a.set:
                k, v = kv.split("=", 1)
                if k == "sex":
                    v = 1 if v.lower() == "male" else 0
                elif k == "exng":
                    v = 1 if v.lower() == "yes" else 0
                elif k in C.NUMERIC or k in ("fbs", "thall"):
                    v = float(v)
                mod[k] = v
            p1 = model.predict_proba(mod)[0, 1]
            print(f"\nWHAT-IF {' '.join(a.set)}:")
            show(a.patient, mod.iloc[0], p1)
            print(f"   >> Probability changed by {100 * (p1 - p0):+.1f} percentage points")
        return

    probs = pd.Series(model.predict_proba(pts[C.FEATURES])[:, 1], index=pts.index)
    for i in pts.sample(a.n, random_state=a.seed).index:
        show(i, pts.loc[i], probs[i], int(pts.loc[i, "actual"]))
    correct = ((probs >= 0.5).astype(int) == pts.actual).sum()
    print("=" * 60)
    print(f"Whole test set: {correct} of {len(pts)} unseen patients classified correctly "
          f"({100 * correct / len(pts):.0f}%).")


if __name__ == "__main__":
    main()
