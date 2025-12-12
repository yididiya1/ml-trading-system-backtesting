
from typing import Tuple

import pandas as pd


def time_split(
    df: pd.DataFrame, train_frac: float, val_frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into train/val/test by time (no shuffling)."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    df_train = df.iloc[:train_end].reset_index(drop=True)
    df_val = df.iloc[train_end:val_end].reset_index(drop=True)
    df_test = df.iloc[val_end:].reset_index(drop=True)

    return df_train, df_val, df_test


def get_feature_target_matrices(
    df: pd.DataFrame, target_col: str = "label"
):
    """Separate features (X) and labels (y).

    Drops non-feature columns like time, OHLCV, and the target column.
    """
    drop_cols = ["time", "open", "high", "low", "close", "volume", target_col]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[target_col].astype(int)

    return X, y
