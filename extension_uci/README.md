# Extension: MI Outcome Risk on Real Hospital Data (UCI)

> This folder is the **extension experiment** of the project. The main experiment (the reference
> paper's dataset) is in the repository root. Run everything below from inside `extension_uci/`.

**UE24CS352A – Machine Learning | Mini-Project, Topic #49 | Section B**
Team: Anshul Poovaiah K (PES1UG24CS071) · Anshdeep Singh Sachdeva (PES1UG24CS070)

Given only the information available **when a heart-attack (MI) patient is admitted to
hospital**, can we predict who is likely to die or develop serious complications, and do
patients fall into natural risk groups?

| | |
|---|---|
| **Data** | UCI *Myocardial infarction complications*: 1,700 patients, 111 clinical features, 12 outcomes |
| **Main task** | Predict in-hospital death (15.9% of patients) |
| **Secondary task** | Predict chronic heart failure (23.2%) + screen all 12 complications |
| **Supervised** | Logistic Regression, k-NN, SVM (RBF), Random Forest, Gradient Boosting |
| **Unsupervised** | PCA, K-Means (+ Ward hierarchical clustering for comparison) |
| **Best result** | Gradient Boosting: **ROC-AUC 0.88** on unseen patients; catches **83% of deaths** at the chosen threshold |

---

## 1. Setup

Download/extract the project ZIP, or clone this repository using its GitHub **Code** button.
Open a terminal in the folder containing the project before the commands below.

Use Python 3.11 and the pinned versions in `../requirements.txt`.

```bash
cd mi-risk-prediction/extension_uci
python3.11 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
python -m pip install -r ../requirements.txt
```

The dataset is already included at `data/raw/MI.data` (see `data/README.md`).

## 2. Run

```bash
python main.py                    # full pipeline, about 1-3 minutes on a laptop
```

Or a single stage at a time:

```bash
python main.py --step eda           # exploratory plots
python main.py --step supervised    # train / tune / evaluate classifiers (death + heart failure)
python main.py --step unsupervised  # PCA + clustering
python main.py --step outcomes      # predictability of all 12 complications
```

Everything is written to:

* `results/figures/` – all plots (numbered in the order they appear in the report)
* `results/metrics/` – CSV/JSON tables of every number in the report
* `models/mortality_model.joblib` – the trained best model (used by the demo)

## 3. Live demo

Run `python main.py --step supervised` from this extension folder first.
Its model is separate from the main project model and is generated locally.

```bash
python demo.py                                   # 5 random unseen test patients
python demo.py --high-risk                       # patients the model rates highest risk
python demo.py --patient 455 --set AGE=85 K_SH_POST=1   # "what-if": change values, see risk move
```

For each patient the demo prints key admission findings, the predicted probability of
death, whether the model flags them, and what actually happened.

## 4. Project structure

```
mi-risk-prediction/
├── main.py                 # runs the whole pipeline
├── demo.py                 # command-line demo on unseen patients
├── requirements.txt
├── data/
│   ├── README.md           # dataset source, licence, format
│   └── raw/MI.data
├── src/
│   ├── config.py           # column names, feature groups, leakage rules, paths
│   ├── data.py             # loading, feature selection, train/test split
│   ├── models.py           # preprocessing pipeline + model definitions + tuning grids
│   ├── eda.py              # exploratory analysis figures
│   ├── supervised.py       # tuning, cross-validation, thresholding, evaluation, importance
│   ├── unsupervised.py     # PCA, K-Means, cluster profiles, cluster-as-feature test
│   └── all_outcomes.py     # screening all 12 complications
├── results/
│   ├── figures/
│   └── metrics/
└── models/
```

## 5. Method

1. **Leakage control.** We predict at admission time, so the 9 columns recorded on
   days 1–3 (pain relapses and repeat drug doses) are removed. Two columns that are >90%
   empty are also dropped, leaving **100 features**.
2. **Preprocessing** (fitted on training data only, inside a scikit-learn `Pipeline`):
   median imputation + "was-missing" flags for lab / blood-pressure values, most-frequent
   imputation for clinical codes, standard scaling.
3. **Split.** Stratified 80/20 train/test (1,360 / 340 patients). Model selection uses training CV. Test results also include exploratory ablations.
4. **Class imbalance.** Only 16% of patients die, so LR, SVM, Random Forest and Gradient Boosting use class weighting; k-NN uses distance weighting. A
   model that always says "survives" would already be 84% accurate, which is why we
   report ROC-AUC, PR-AUC, recall and precision instead of accuracy.
5. **Model selection.** Grid search with 5-fold stratified cross-validation on the
   training set, scored by ROC-AUC. The best model is chosen by CV score, not test score.
6. **Decision threshold.** Missing a patient who will die is worse than a false alarm,
   so a high-sensitivity threshold (target recall ≥ 80%) is chosen from out-of-fold
   training predictions.
7. **Interpretation.** Permutation importance on the test set.
8. **Unsupervised.** PCA (100 features → 10 components), K-Means with k chosen by
   silhouette score, cluster profiles compared against outcomes (labels are never used to
   form clusters), and a test of whether cluster membership improves the classifier.

## 6. Results

Regenerated with Python 3.11 and the pinned dependencies in `../requirements.txt`.

### Predicting death (test set, 340 unseen patients, 54 deaths)

| Model | CV ROC-AUC | Test ROC-AUC | Recall @0.5 | Precision @0.5 |
|---|---|---|---|---|
| **Gradient Boosting** | 0.899 ± 0.010 | 0.882 | 0.67 | 0.45 |
| Random Forest | 0.897 ± 0.009 | 0.880 | 0.20 | 1.00 |
| Logistic Regression | 0.883 ± 0.013 | 0.858 | 0.72 | 0.44 |
| SVM (RBF) | 0.881 ± 0.018 | 0.874 | 0.44 | 0.56 |
| k-NN | 0.838 ± 0.025 | 0.828 | 0.04 | 1.00 |

With the high-sensitivity threshold (0.38), Gradient Boosting catches **45 of 54 deaths
(recall 0.83)** with specificity 0.78 and precision 0.42.

Most important admission features: ICU systolic blood pressure, time from pain onset to
hospital, age, anterior MI on ECG, recent coronary heart disease.

### Predicting chronic heart failure
Harder task: best model Random Forest, CV ROC-AUC 0.73, test ROC-AUC 0.72.

### Unsupervised findings
* The data is high-dimensional and diffuse: 80 principal components are needed for 90%
  of the variance.
* K-Means (k = 3) finds groups with clearly different death rates (12%, 21%, 35%;
  χ² p < 1e-10), even though outcomes were never shown to the algorithm. The highest-risk
  cluster is older, has more prior heart failure and lower blood pressure.
* Adding cluster membership as a feature did **not** improve prediction (CV AUC 0.880 →
  0.879): the supervised model already captures that information directly.

### All 12 complications
Death is by far the most predictable outcome (CV AUC 0.89). Rare complications such as
supraventricular tachycardia (20 cases) are close to chance level.

## 7. Limitations
* Single hospital, 1992–1995 data, so it may not reflect modern treatment.
* Remaining in-ICU drug features may be unavailable at the intended prediction time;
  the admission-only interpretation has not been fully validated.
* The near-empty-column filter uses missingness across the full dataset before splitting.
* Cluster count is chosen on the full dataset before the exploratory cluster-feature comparison.
* Small number of positive cases for most complications.
* The model supports, not replaces, clinical judgement.

## Reference
Golovenkin S.E. et al. *Myocardial infarction complications* [Dataset]. UCI Machine
Learning Repository, 2020. https://archive.ics.uci.edu/dataset/579
