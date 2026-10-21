"""Preprocessing pipeline and the set of supervised models we compare."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config as C
from .data import categorical_cols, continuous_cols


def make_preprocessor(feats):
    """Impute + scale.

    * Continuous lab / blood-pressure values -> median imputation, plus a binary
      "was missing" flag (missingness itself can be informative in hospitals).
    * Binary / ordinal clinical codes -> most-frequent imputation.
    * Everything is then standardised (needed by LR, SVM, kNN; harmless for trees).
    All statistics are learned on the training fold only (no test leakage).
    """
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        [("num", num, continuous_cols(feats)), ("cat", cat, categorical_cols(feats))],
        verbose_feature_names_out=False,
    )


def model_zoo():
    """Candidate models. class_weight='balanced' up-weights the rare positive class
    (deaths are ~16% of patients) so the model is not rewarded for always saying 'survives'."""
    rs = C.RANDOM_STATE
    return {
        "Logistic Regression": LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=5000),
        "k-Nearest Neighbours": KNeighborsClassifier(n_neighbors=25, weights="distance"),
        "SVM (RBF kernel)": SVC(
            C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=rs),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample",
            random_state=rs, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
            class_weight="balanced", random_state=rs),
    }


# Small hyper-parameter grids, searched with stratified CV on the training set only.
PARAM_GRIDS = {
    "Logistic Regression": {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0]},
    "k-Nearest Neighbours": {"clf__n_neighbors": [11, 25, 51]},
    "SVM (RBF kernel)": {"clf__C": [0.3, 1.0, 3.0], "clf__gamma": ["scale", 0.003]},
    "Random Forest": {"clf__min_samples_leaf": [1, 3, 8], "clf__max_features": ["sqrt", 0.3]},
    "Gradient Boosting": {"clf__learning_rate": [0.03, 0.1], "clf__max_leaf_nodes": [7, 15]},
}


def make_pipeline(feats, clf):
    return Pipeline([("prep", make_preprocessor(feats)), ("clf", clf)])
