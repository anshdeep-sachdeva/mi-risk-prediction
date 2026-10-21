"""Configuration for the main experiment: Heart Attack Analysis & Prediction dataset.

Same 303 patients as the Kaggle dataset used in the reference CS229 paper
(Shi, Bechler & Ho, "Predicting myocardial infarction risk using supervised and
unsupervised machine learning methods"). This copy stores categories as readable labels.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "heart.csv"
FIG_DIR = ROOT / "results" / "figures"
METRICS_DIR = ROOT / "results" / "metrics"
MODEL_DIR = ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS, CV_REPEATS = 5, 3          # repeated stratified 5-fold CV (15 folds)
TARGET = "target"                     # 1 = "high MI risk" label, as in the reference paper

NUMERIC = ["age", "trtbps", "chol", "thalachh", "oldpeak", "ca"]
BINARY = ["sex", "fbs", "exng"]       # encoded 0/1 (sex: 1 = male, exng: 1 = yes)
CATEGORICAL = ["cp", "rest_ecg", "slope", "thall"]   # one-hot encoded
FEATURES = NUMERIC + BINARY + CATEGORICAL

THALL_NAMES = {1: "Fixed defect", 2: "Normal", 3: "Reversible defect"}

NAMES = {
    "age": "Age", "sex": "Sex (male)", "cp": "Chest pain type", "trtbps": "Resting blood pressure",
    "chol": "Cholesterol", "fbs": "Fasting blood sugar > 120", "rest_ecg": "Resting ECG",
    "thalachh": "Max heart rate", "exng": "Exercise-induced angina", "oldpeak": "ST depression (oldpeak)",
    "slope": "ST slope", "ca": "No. of major vessels (ca)", "thall": "Thallium test",
}

# Reference paper's reported TEST-set results (Table 2) for comparison.
PAPER_TEST = {
    "Logistic Regression": {"accuracy": 0.87, "roc_auc": 0.86, "precision": 0.84, "recall": 0.94, "f1": 0.89},
    "Gradient Boosting":   {"accuracy": 0.84, "roc_auc": 0.83, "precision": 0.81, "recall": 0.91, "f1": 0.86},
    "Bagging":             {"accuracy": 0.80, "roc_auc": 0.80, "precision": 0.78, "recall": 0.88, "f1": 0.83},
    "Random Forest":       {"accuracy": 0.79, "roc_auc": 0.78, "precision": 0.78, "recall": 0.85, "f1": 0.81},
    "SVM (RBF kernel)":    {"accuracy": 0.46, "roc_auc": 0.50, "precision": 0.00, "recall": 0.00, "f1": 0.00},
}
PAPER_ARI = {"K-Means": 0.02, "DBSCAN": 0.06}


def pretty(name: str) -> str:
    """Readable name for an encoded feature, e.g. 'cp_Typical Angina' -> 'Chest pain: Typical Angina'."""
    for c in CATEGORICAL:
        if name.startswith(c + "_"):
            val = name[len(c) + 1:]
            if c == "thall":
                val = THALL_NAMES.get(int(float(val)), val)
            return f"{NAMES[c]}: {val}"
    return NAMES.get(name, name)
