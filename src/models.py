"""Preprocessing and the nine model families compared in the reference paper."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config as C


def make_preprocessor(scale=True):
    """Numeric -> standardised; binary -> kept as 0/1; categorical -> one-hot.
    Inside a Pipeline, so scaling statistics come from the training folds only."""
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [("num", num, C.NUMERIC),
         ("bin", "passthrough", C.BINARY),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), C.CATEGORICAL)],
        verbose_feature_names_out=False,
    )


def pipe(clf):
    return Pipeline([("prep", make_preprocessor()), ("clf", clf)])


def model_zoo():
    rs = C.RANDOM_STATE
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000),
            {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]}),
        "Elastic Net": (
            LogisticRegression(solver="saga", penalty="elasticnet", l1_ratio=0.5,
                               max_iter=20000, random_state=rs),
            {"clf__C": [0.03, 0.1, 0.3, 1, 3], "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=rs),
            {"clf__learning_rate": [0.01, 0.05, 0.1], "clf__n_estimators": [100, 300],
             "clf__max_depth": [1, 2, 3]}),
        "Bagging": (
            BaggingClassifier(DecisionTreeClassifier(random_state=rs), random_state=rs),
            {"clf__n_estimators": [25, 50, 100, 300]}),
        "Random Forest": (
            RandomForestClassifier(n_estimators=500, random_state=rs),
            {"clf__max_depth": [None, 4], "clf__min_samples_leaf": [1, 5],
             "clf__max_features": ["sqrt", 0.5]}),
        "SVM (linear kernel)": (
            SVC(kernel="linear", probability=True, random_state=rs),
            {"clf__C": [0.001, 0.01, 0.1, 1]}),
        "SVM (RBF kernel)": (
            SVC(kernel="rbf", probability=True, random_state=rs),
            {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1]}),
        "SVM (polynomial kernel)": (
            SVC(kernel="poly", probability=True, random_state=rs),
            {"clf__C": [0.01, 0.1, 1], "clf__degree": [2, 3]}),
        "Neural Network": (
            # same architecture as the paper: 4 hidden ReLU layers of 18, 15, 10, 8 units
            MLPClassifier(hidden_layer_sizes=(18, 15, 10, 8), activation="relu", max_iter=3000,
                          early_stopping=True, validation_fraction=0.2, random_state=rs),
            {"clf__alpha": [1e-3, 1e-1, 1.0]}),
    }
