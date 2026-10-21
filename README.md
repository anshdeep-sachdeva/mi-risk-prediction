# Predicting Myocardial Infarction Risk with Supervised and Unsupervised Learning

**UE24CS352A – Machine Learning | Mini-Project, Topic #49 | Section B**
Team: Anshul Poovaiah K (PES1UG24CS071) · Anshdeep Singh Sachdeva (PES1UG24CS070)

We reproduce and extend the reference study *"Predicting myocardial infarction risk using
supervised and unsupervised machine learning methods"* (Shi, Bechler & Ho, Stanford CS229):
on a supplied, preprocessed copy of the 303-patient dataset, we cluster patients and
compare nine classifier families to predict each patient's MI-risk label.

| | |
|---|---|
| **Main dataset** | Heart Attack Analysis & Prediction (Kaggle): 303 patients, 13 features, label 1: 54.5% (see label caveat below) |
| **Supervised** | Logistic Regression, Elastic Net, Gradient Boosting, Bagging, Random Forest, SVM (linear / RBF / polynomial), Neural Network |
| **Unsupervised** | PCA, K-Means, DBSCAN, hierarchical (Ward) clustering, evaluated with the Adjusted Rand Index |
| **Best model** | Logistic Regression: test ROC-AUC **0.91** (95% CI 0.83–0.97), recall **0.94**, accuracy 0.85 |
| **Key finding** | Standardising features raises clustering agreement with the label from ARI 0.02 (paper) to **0.30** |
| **Extension** | Same approach on a larger, real hospital dataset (1,700 MI patients): see `extension_uci/` |

---

## 1. Setup

Download/extract the project ZIP, or clone this repository using its GitHub **Code** button.
Open a terminal in the folder containing the project before the commands below.

Use Python 3.11 and the pinned versions in `requirements.txt`.

```bash
cd mi-risk-prediction
python3.11 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Both datasets are included (see `data/README.md` and `extension_uci/data/README.md`).

## 2. Run

```bash
python main.py                      # main experiment, about 1-3 minutes
python main.py --step eda           # or: supervised | unsupervised
```

Outputs go to `results/figures/`, `results/metrics/` and `models/`.

## 3. Live demo

Run `python main.py --step supervised` first. The trained model and demo patients
are generated locally in `models/` and are intentionally excluded from Git.

```bash
python demo.py                                        # 5 random unseen test patients
python demo.py --patient 179                          # one patient
python demo.py --patient 179 --set thalachh=120 exng=Yes   # what-if
```

## 4. Project structure

```
mi-risk-prediction/
├── main.py                 # runs the main experiment
├── demo.py                 # command-line demo on unseen patients
├── requirements.txt
├── data/raw/heart.csv      # main dataset (+ data/README.md)
├── src/
│   ├── config.py           # columns, feature groups, paths, reference-paper numbers
│   ├── data.py             # loading, encoding, train/test split
│   ├── models.py           # preprocessing pipeline + the 9 model families and tuning grids
│   ├── eda.py              # exploratory figures
│   ├── supervised.py       # repeated-CV tuning, test evaluation, bootstrap CI, interpretation
│   └── unsupervised.py     # PCA, K-Means, DBSCAN, Ward, ARI; unscaled vs standardised
├── results/{figures,metrics}/
├── models/
└── extension_uci/          # extension: 1,700 hospitalised MI patients (own main.py, demo.py, README)
```

## 5. Method

1. **Encoding.** Numeric features (age, blood pressure, cholesterol, max heart rate, ST
   depression, vessel count) are standardised; sex, fasting blood sugar and exercise angina are
   0/1; chest pain type, resting ECG, ST slope and thallium result are one-hot encoded. All of
   this sits inside a scikit-learn `Pipeline`, so it is learned from training data only.
2. **Split.** Stratified 80/20: 242 training and 61 test patients.
3. **Tuning and selection.** Grid search with **repeated stratified 5-fold CV (3 repeats = 15
   folds)** on the training set, scored by ROC-AUC (F1 also recorded). The best model is chosen by
   CV score. The paper also used CV for some models, then selected candidates using a separate 20% validation set (~60 patients).
4. **Evaluation.** Tuned models are evaluated on the held-out test set; model selection uses training CV scores. Because it is small, we also report a 95% bootstrap
   confidence interval for ROC-AUC.
5. **Interpretation.** Logistic-regression odds ratios and permutation importance.
6. **Unsupervised.** PCA, then K-Means (k = 2) over 1–10 components, DBSCAN over a range of eps,
   and Ward clustering, each scored by ARI against the label. Every analysis is run twice:
   **unscaled** (paper-inspired baseline) and **standardised + one-hot**.
   PCA component counts and DBSCAN epsilon are selected by ARI against the full dataset
   labels. These are exploratory, label-guided comparisons, not independent validation
   of entirely label-free model selection.

## 6. Results

Regenerated with Python 3.11 and the pinned dependencies in `requirements.txt`.

### Supervised (test set: 61 unseen patients)

| Model | CV ROC-AUC | Test ROC-AUC | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.908 ± 0.037** | **0.908** | **0.85** | **0.94** | **0.82** | **0.87** |
| Random Forest | 0.907 ± 0.040 | 0.903 | 0.77 | 0.88 | 0.74 | 0.81 |
| Elastic Net | 0.906 ± 0.037 | 0.908 | 0.85 | 0.94 | 0.82 | 0.87 |
| SVM (linear) | 0.905 ± 0.042 | 0.918 | 0.84 | 0.94 | 0.79 | 0.86 |
| SVM (RBF) | 0.904 ± 0.038 | 0.910 | 0.84 | 0.94 | 0.79 | 0.86 |
| Gradient Boosting | 0.896 ± 0.046 | 0.880 | 0.79 | 0.88 | 0.76 | 0.82 |
| SVM (polynomial) | 0.891 ± 0.040 | 0.913 | 0.82 | 0.94 | 0.78 | 0.85 |
| Bagging | 0.882 ± 0.050 | 0.859 | 0.82 | 0.94 | 0.78 | 0.85 |
| Neural Network | 0.831 ± 0.056 | 0.840 | 0.72 | 0.91 | 0.68 | 0.78 |

The leading CV scores are close. Logistic regression is a simple, interpretable choice;
overlapping fold standard deviations do not establish statistical equivalence.

### Comparison with the reference paper (test set)

| Model | Paper AUC | Our AUC | Paper recall | Our recall |
|---|---|---|---|---|
| Logistic Regression | 0.86 | **0.91** | 0.94 | **0.94** |
| Gradient Boosting | 0.83 | **0.88** | 0.91 | 0.88 |
| Bagging | 0.80 | **0.86** | 0.88 | **0.94** |
| Random Forest | 0.78 | **0.90** | 0.85 | **0.88** |
| SVM (RBF) | 0.50 | **0.91** | 0.00 | **0.94** |

Logistic regression is selected by training CV in this experiment. These published-paper
comparisons are descriptive: the split, training size, preprocessing and tuning differ.
They do not establish that our methods outperform the paper on identical test patients.
The paper already used min-max scaling for SVM; its weak RBF result cannot be attributed
to missing scaling alone.

### Unsupervised

| Method | Paper ARI | Ours, unscaled | Ours, standardised |
|---|---|---|---|
| K-Means (k = 2) | 0.02 | 0.02 | **0.30** |
| DBSCAN | 0.06 | 0.05 | **0.15** |
| Ward hierarchical | – | – | **0.29** |

In our unscaled run, PC1 explains 75% of variance and has an absolute cholesterol
loading of about **0.998**. Scaling changes the feature weighting and improves ARI
in this experiment; this does not prove the cause of the original paper's findings.
The selected K-Means groups have 25% vs 80% label-1 patients.

## 7. Data caveat
The supplied labels are retained for comparison with the paper. “High risk” in existing
plots is the paper's name for label 1, not a verified future-heart-attack outcome.
The original Cleveland target describes existing heart-disease status; the mapping of
this supplied copy to that original target has not been verified. Predictions should
therefore be interpreted as dataset-label probabilities, not clinical risk estimates.
See `data/README.md` for the known preprocessing provenance gap.

## 8. Extension: real hospital data (`extension_uci/`)
The same ideas applied to 1,700 hospitalised MI patients from the UCI *Myocardial infarction
complications* dataset (111 features, missing values, 16% mortality). Gradient Boosting predicts
in-hospital death with test ROC-AUC 0.88 and catches 83% of deaths at a tuned threshold; K-Means
finds patient groups with 12% / 21% / 35% mortality. See `extension_uci/README.md`.

## 9. Limitations
Small dataset (61 test patients, so wide confidence intervals); single source; the label caveat
above; no external validation.

## References
1. Shi K., Bechler K., Ho V. *Predicting myocardial infarction risk using supervised and
   unsupervised machine learning methods.* Stanford CS229 project report, 2021. https://cs229.stanford.edu/proj2021spr/report2/81980162.pdf
2. Rahman R. *Heart Attack Analysis & Prediction Dataset.* Kaggle. https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
3. Golovenkin S.E. et al. *Myocardial infarction complications* [Dataset]. UCI Machine Learning
   Repository, 2020. https://archive.ics.uci.edu/dataset/579
4. UCI Heart Disease: original target and feature definitions. https://archive.ics.uci.edu/dataset/45/heart+disease
