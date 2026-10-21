"""Loading, encoding and splitting the heart dataset."""
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def load() -> pd.DataFrame:
    if not C.DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {C.DATA_PATH} (see data/README.md).")
    df = pd.read_csv(C.DATA_PATH)
    df["sex"] = (df["sex"].astype(str).str.strip().str.lower() == "male").astype(int)
    df["fbs"] = df["fbs"].astype(str).str.strip().str.upper().isin(["TRUE", "1"]).astype(int)
    df["exng"] = (df["exng"].astype(str).str.strip().str.lower() == "yes").astype(int)
    for c in ["cp", "rest_ecg", "slope"]:
        df[c] = df[c].astype(str).str.strip()
    df["thall"] = df["thall"].astype(int)
    return df


def get_xy(df=None):
    df = load() if df is None else df
    return df[C.FEATURES].copy(), df[C.TARGET].astype(int)


def split(X, y):
    return train_test_split(X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)


def encoded(X, scale=True):
    """One-hot encoded (and optionally standardised) matrix with readable column names."""
    from .models import make_preprocessor
    pre = make_preprocessor(scale=scale)
    Z = pre.fit_transform(X)
    return pd.DataFrame(Z, columns=pre.get_feature_names_out(), index=X.index)
