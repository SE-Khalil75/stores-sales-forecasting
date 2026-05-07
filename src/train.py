"""Train the LightGBM regressor and save the artifact.

Usage: python -m src.train

This is the production path that mirrors notebooks/01_train_and_evaluate.ipynb:
same feature set, same split, same hyperparameters. Both write the same
artifact at models/model.pkl which the Streamlit app consumes.
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from src.evaluate import compute_metrics, time_based_split
from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    LAG_COLUMNS,
    build_features,
)

DATA_PATH = Path("data/stores-sales.csv")
MODEL_PATH = Path("models/model.pkl")

TRAIN_TEST_CUTOFF = "2012-06-01"
# Inner cutoff: last ~8 weeks of train period serve as val for early stopping.
TRAIN_VAL_CUTOFF = "2012-04-01"

LGBM_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=20,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    random_state=42,
    verbose=-1,
)


def main() -> None:
    print(f"Loading {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH)
    features_df = build_features(raw)  # drops warm-up rows by default

    train_full, test = time_based_split(features_df, TRAIN_TEST_CUTOFF)
    train_inner, val = time_based_split(train_full, TRAIN_VAL_CUTOFF)

    print(
        f"Rows  train={len(train_full)} (inner={len(train_inner)}, val={len(val)})"
        f"  test={len(test)}"
    )
    print(
        f"Dates train={train_full['date'].min().date()}..{train_full['date'].max().date()}"
        f"  test={test['date'].min().date()}..{test['date'].max().date()}"
    )

    X_inner, y_inner = train_inner[FEATURE_COLUMNS], train_inner["weekly_sales"]
    X_val, y_val = val[FEATURE_COLUMNS], val["weekly_sales"]
    X_train_full = train_full[FEATURE_COLUMNS]
    y_train_full = train_full["weekly_sales"]
    X_test, y_test = test[FEATURE_COLUMNS], test["weekly_sales"]

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_inner,
        y_inner,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    train_metrics = compute_metrics(y_train_full, model.predict(X_train_full))
    test_metrics = compute_metrics(y_test, model.predict(X_test))

    print("\n=== Metrics ===")
    print(
        f"Train  RMSE={train_metrics['rmse']:>12,.2f}"
        f"  MAE={train_metrics['mae']:>10,.2f}"
        f"  MAPE={train_metrics['mape']:.4f}"
    )
    print(
        f"Test   RMSE={test_metrics['rmse']:>12,.2f}"
        f"  MAE={test_metrics['mae']:>10,.2f}"
        f"  MAPE={test_metrics['mape']:.4f}"
    )

    importances = (
        pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
        .sort_values(ascending=False)
    )
    print("\n=== Top 15 feature importances ===")
    print(importances.head(15).to_string())

    # Per-store fallback stats — used by the app to fill NaN lag values
    # for uploads that don't carry full per-store history (e.g. a single-row
    # upload). Computed on training data only.
    store_stats = (
        train_full.groupby("store")["weekly_sales"].agg(["mean", "std"])
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "lag_columns": LAG_COLUMNS,
        "store_stats": store_stats,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_test_cutoff": TRAIN_TEST_CUTOFF,
        "best_iteration": (
            int(model.best_iteration_) if model.best_iteration_ else None
        ),
        "n_train_rows": int(len(train_full)),
        "n_test_rows": int(len(test)),
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nSaved model artifact to {MODEL_PATH}")


if __name__ == "__main__":
    main()
