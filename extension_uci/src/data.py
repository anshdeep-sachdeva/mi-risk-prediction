"""Loading the raw UCI file and building the feature matrix / targets."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def load_raw() -> pd.DataFrame:
    """Read MI.data (no header, '?' = missing) and attach the official column names."""
    if not C.DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {C.DATA_PATH}.\n"
            "Download 'Myocardial infarction complications' from the UCI repository "
            "(https://archive.ics.uci.edu/dataset/579) and place MI.data in data/raw/."
        )
    df = pd.read_csv(C.DATA_PATH, header=None, na_values="?", names=C.COLUMNS)
    return df


def select_features(df: pd.DataFrame, verbose: bool = True) -> list:
    """Admission-time input features, after removing leaky and near-empty columns."""
    feats = [c for c in C.INPUTS if c not in C.LATER_THAN_ADMISSION]
    miss = df[feats].isna().mean()
    too_sparse = miss[miss > C.MOSTLY_MISSING_THRESHOLD].index.tolist()
    feats = [c for c in feats if c not in too_sparse]
    if verbose:
        print(f"  dropped {len(C.LATER_THAN_ADMISSION)} post-admission columns (leakage)")
        print(f"  dropped {len(too_sparse)} near-empty columns: {too_sparse}")
        print(f"  using {len(feats)} admission-time features")
    return feats


def make_target(df: pd.DataFrame, target: str) -> pd.Series:
    """Binary target. LET_IS is coded 0 = alive, 1-7 = cause of death -> 1 = died."""
    return (df[target] > 0).astype(int).rename(target)


def get_dataset(target: str = C.PRIMARY_TARGET, verbose: bool = True):
    df = load_raw()
    feats = select_features(df, verbose=verbose)
    X = df[feats].copy()
    y = make_target(df, target)
    return X, y, feats


def split(X, y):
    """Stratified 80/20 split so both sets keep the same death rate."""
    return train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE
    )


def continuous_cols(feats):
    return [c for c in feats if c in C.CONTINUOUS]


def categorical_cols(feats):
    return [c for c in feats if c not in C.CONTINUOUS]
