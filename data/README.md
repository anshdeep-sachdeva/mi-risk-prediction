# Data

`raw/heart.csv` — **Heart Attack Analysis & Prediction Dataset** (303 patients, 13 features,
binary label `target`: 1 = high MI risk, 0 = low MI risk, as used in the reference paper).

* Kaggle (Rahman R.): https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
* Derived from the Cleveland subset of the UCI Heart Disease database.
* This supplied copy stores categories as readable labels (e.g. `Male`, `Typical Angina`).
  It has 303 rows, label counts 165 / 138, and no `ca = 4` or `thall = 0` values.
  The original handoff described those codes as corrected, but supplied no original-to-final
  mapping or cleaning script. We cannot verify which rows or values changed. The current
  loader does not perform those corrections; it reads this supplied CSV unchanged.
* No missing values.

| Column | Meaning |
|---|---|
| age | Age (years) |
| sex | Male / Female |
| cp | Chest pain type |
| trtbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| rest_ecg | Resting ECG result |
| thalachh | Maximum heart rate achieved |
| exng | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak-exercise ST segment |
| ca | Number of major vessels coloured by fluoroscopy (0-3) |
| thall | Thallium stress test (1 fixed defect, 2 normal, 3 reversible defect) |
| target | 1 = high MI risk, 0 = low MI risk |

The extension experiment uses a second dataset, stored in `extension_uci/data/`.

Label names follow the reference paper; their mapping to the original Cleveland
heart-disease target is unverified. They are not validated future-MI risk labels.

SHA-256 of the supplied `raw/heart.csv` (for reproducibility):
`f490cc5bf5ee5920b44f136bb58fb3f0c17b7bc921a06118e81af7d5e0fd03e8`
