"""Predicting myocardial infarction risk with supervised and unsupervised ML.

Main experiment on the Heart Attack Analysis & Prediction dataset (303 patients),
the dataset used by the reference paper.

Usage:
    python main.py                      # everything (about 1-3 minutes)
    python main.py --step eda | supervised | unsupervised
The extension experiment on the UCI MI-complications dataset lives in extension_uci/.
"""
import argparse
import json
import time

from src import config as C, eda, supervised, unsupervised


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--step", choices=["all", "eda", "supervised", "unsupervised"], default="all")
    step = ap.parse_args().step
    for d in (C.FIG_DIR, C.METRICS_DIR, C.MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if step in ("all", "eda"):
        print("\n[1/3] Exploratory data analysis")
        s = eda.run()
        json.dump(s, open(C.METRICS_DIR / "eda_summary.json", "w"), indent=2)
        print(f"  {s['patients']} patients, {s['high_risk']} labelled high risk ({s['high_risk_pct']}%), "
              f"{s['missing_values']} missing values")
    if step in ("all", "supervised"):
        print("\n[2/3] Supervised learning: 9 model families")
        supervised.run()
    if step in ("all", "unsupervised"):
        print("\n[3/3] Unsupervised learning: PCA, K-Means, DBSCAN, Ward")
        unsupervised.run()
    print(f"\nDone in {time.time() - t0:.0f}s. Results in results/figures and results/metrics.")


if __name__ == "__main__":
    main()
