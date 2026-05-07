"""Time-based split and regression metrics."""

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)


def time_based_split(
    df: pd.DataFrame, cutoff_date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically. Rows with date < cutoff go to train, rest to test.

    `df` must contain a datetime-parsed `date` column.
    """
    cutoff = pd.Timestamp(cutoff_date)
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    return train, test


def compute_metrics(y_true, y_pred) -> dict:
    """Return RMSE, MAE, and MAPE as floats."""
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }
