"""Train the LightGBM regressor and save the artifact.

Usage:
  python -m src.train             # full feature set (lag + rolling)
  python -m src.train --no-lag    # no lag/rolling features

The --no-lag variant produces a separate artifact at models/model_no_lag.pkl
so the Streamlit app can switch between the two.
"""

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from src.evaluate import compute_metrics, time_based_split
from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    LAG_COLUMNS,
    NUMERIC_FEATURES,
    build_features,
)

DATA_PATH = Path("data/stores-sales.csv")
MODEL_PATH = Path("models/model.pkl")
MODEL_PATH_NO_LAG = Path("models/model_no_lag.pkl")

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-lag",
        action="store_true",
        help="Train without lag/rolling features. Saves to models/model_no_lag.pkl.",
    )
    args = parser.parse_args()

    feature_columns = (
        ["store"] + NUMERIC_FEATURES if args.no_lag else FEATURE_COLUMNS
    )
    model_path = MODEL_PATH_NO_LAG if args.no_lag else MODEL_PATH
    # No-lag model has no NaN-producing features, so we keep warm-up rows.
    drop_warmup = not args.no_lag

    print(f"Loading {DATA_PATH}  (variant={'no-lag' if args.no_lag else 'full'})")
    raw = pd.read_csv(DATA_PATH)
    features_df = build_features(raw, drop_warmup=drop_warmup)

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

    X_inner, y_inner = train_inner[feature_columns], train_inner["weekly_sales"]
    X_val, y_val = val[feature_columns], val["weekly_sales"]
    X_train_full = train_full[feature_columns]
    y_train_full = train_full["weekly_sales"]
    X_test, y_test = test[feature_columns], test["weekly_sales"]

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
        pd.Series(model.feature_importances_, index=feature_columns)
        .sort_values(ascending=False)
    )
    print("\n=== Top 15 feature importances ===")
    print(importances.head(15).to_string())

    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "categorical_features": CATEGORICAL_FEATURES,
        "uses_lag_features": not args.no_lag,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_test_cutoff": TRAIN_TEST_CUTOFF,
        "best_iteration": (
            int(model.best_iteration_) if model.best_iteration_ else None
        ),
        "n_train_rows": int(len(train_full)),
        "n_test_rows": int(len(test)),
    }
    if not args.no_lag:
        # Per-store fallback stats — used by the app to fill NaN lag values
        # for uploads that don't carry full per-store history (e.g. a single-row
        # upload). Only the lag-feature model needs them.
        artifact["lag_columns"] = LAG_COLUMNS
        artifact["store_stats"] = (
            train_full.groupby("store")["weekly_sales"].agg(["mean", "std"])
        )

    joblib.dump(artifact, model_path)
    print(f"\nSaved model artifact to {model_path}")


if __name__ == "__main__":
    main()
