"""Run the full pipeline: EDA -> supervised models -> unsupervised analysis -> all outcomes.

Usage:
    python main.py              # everything (~2-4 minutes on a laptop)
    python main.py --step eda   # or: supervised | unsupervised | outcomes
Outputs go to results/figures, results/metrics and models/.
"""
import argparse
import json
import time

from src import all_outcomes, config as C, eda, supervised, unsupervised


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--step", choices=["all", "eda", "supervised", "unsupervised", "outcomes"],
                    default="all")
    step = ap.parse_args().step
    t0 = time.time()
    if step in ("all", "eda"):
        print("\n[1/5] Exploratory data analysis")
        s = eda.run()
        json.dump(s, open(C.METRICS_DIR / "eda_summary.json", "w"), indent=2)
        print(f"  {s['patients']} patients, {s['deaths']} deaths ({s['death_rate_pct']}%)")
    if step in ("all", "supervised"):
        print("\n[2/5] Supervised: predicting death (LET_IS)")
        supervised.run(C.PRIMARY_TARGET, "mortality")
        print("\n[3/5] Supervised: predicting chronic heart failure (ZSN)")
        supervised.run(C.SECONDARY_TARGET, "heart_failure", verbose=False)
    if step in ("all", "unsupervised"):
        print("\n[4/5] Unsupervised: PCA + clustering")
        unsupervised.run()
    if step in ("all", "outcomes"):
        print("\n[5/5] Screening all 12 complications")
        all_outcomes.run()
    print(f"\nDone in {time.time() - t0:.0f}s. See results/ for figures and metrics.")


if __name__ == "__main__":
    for d in (C.FIG_DIR, C.METRICS_DIR, C.MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    main()
