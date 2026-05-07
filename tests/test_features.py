"""Tests for src.features."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS,
    LAG_COLUMNS,
    build_features,
    fill_warmup_with_fallback,
)


def _history(
    start: str = "05-02-2010", n_weeks: int = 20, store: int = 1
) -> pd.DataFrame:
    """Synthetic weekly history for one store. Dates in DD-MM-YYYY."""
    start_ts = pd.to_datetime(start, dayfirst=True)
    dates = pd.date_range(start_ts, periods=n_weeks, freq="7D")
    return pd.DataFrame(
        {
            "store": store,
            "date": dates.strftime("%d-%m-%Y"),
            "weekly_sales": 1_000_000.0 + 10_000.0 * np.arange(n_weeks),
            "holiday_flag": 0,
            "temperature": 60.0,
            "fuel_Price": 3.0,
            "cpi": 200.0,
            "unemployment": 7.0,
        }
    )


def test_date_parsing_uses_dayfirst():
    """05-02-2010 must parse as Feb 5, not May 2."""
    out = build_features(_history(start="05-02-2010", n_weeks=20), drop_warmup=False)
    out = out.sort_values("date").reset_index(drop=True)
    assert out["date"].iloc[0] == pd.Timestamp("2010-02-05")
    assert out["month"].iloc[0] == 2
    assert out["day_of_year"].iloc[0] == 36


def test_super_bowl_flag():
    out = build_features(_history(start="05-02-2010", n_weeks=20), drop_warmup=False)
    sb = out[out["date"] == pd.Timestamp("2010-02-12")]
    assert sb["is_super_bowl"].iloc[0] == 1
    assert sb["is_pre_super_bowl"].iloc[0] == 0


def test_thanksgiving_flag():
    out = build_features(_history(start="05-11-2010", n_weeks=10), drop_warmup=False)
    tg = out[out["date"] == pd.Timestamp("2010-11-26")]
    assert tg["is_thanksgiving"].iloc[0] == 1


def test_christmas_and_pre_christmas_flags():
    out = build_features(_history(start="03-12-2010", n_weeks=10), drop_warmup=False)
    xmas = out[out["date"] == pd.Timestamp("2010-12-31")]
    pre = out[out["date"] == pd.Timestamp("2010-12-24")]
    assert xmas["is_christmas"].iloc[0] == 1
    assert pre["is_pre_christmas"].iloc[0] == 1


def test_lag_features_compute_correctly():
    out = build_features(_history(n_weeks=20), drop_warmup=False)
    out = out.sort_values("date").reset_index(drop=True)
    # weekly_sales[i] = 1_000_000 + 10_000*i, so lag_1[i] should be that of i-1.
    assert pd.isna(out["lag_1_sales"].iloc[0])
    assert out["lag_1_sales"].iloc[1] == 1_000_000.0
    assert out["lag_1_sales"].iloc[5] == 1_040_000.0
    assert out["lag_4_sales"].iloc[5] == 1_010_000.0


def test_lag_features_isolated_per_store():
    """Each store's first row must have NaN lag_1, never inherit from another store."""
    df = pd.concat(
        [
            _history(n_weeks=15, store=1),
            _history(n_weeks=15, store=2),
        ],
        ignore_index=True,
    )
    out = build_features(df, drop_warmup=False).sort_values(["store", "date"])
    first_per_store = out.groupby("store").head(1)
    assert first_per_store["lag_1_sales"].isna().all()


def test_drop_warmup_removes_nan_rows():
    out = build_features(_history(n_weeks=20))  # drop_warmup=True default
    assert out[LAG_COLUMNS].notna().all().all()
    # rolling_mean_12 is the longest warmup; 12 rows dropped from a 20-row series.
    assert len(out) == 8


def test_cyclical_encoding_on_unit_circle():
    out = build_features(_history(n_weeks=20), drop_warmup=False)
    np.testing.assert_allclose(
        out["week_sin"] ** 2 + out["week_cos"] ** 2, 1.0, atol=1e-10
    )
    np.testing.assert_allclose(
        out["month_sin"] ** 2 + out["month_cos"] ** 2, 1.0, atol=1e-10
    )


def test_all_feature_columns_present():
    out = build_features(_history(n_weeks=20))
    for col in FEATURE_COLUMNS:
        assert col in out.columns, f"missing column: {col}"


def test_no_nans_after_drop_warmup():
    out = build_features(_history(n_weeks=20))
    assert out[FEATURE_COLUMNS].notna().all().all()


def test_fuel_price_renamed_from_capital_p():
    out = build_features(_history(n_weeks=20), drop_warmup=False)
    assert "fuel_price" in out.columns
    assert "fuel_Price" not in out.columns


def test_accepts_already_lowercase_fuel_price():
    df = _history(n_weeks=20).rename(columns={"fuel_Price": "fuel_price"})
    out = build_features(df, drop_warmup=False)
    assert "fuel_price" in out.columns


def test_store_dtype_is_integer():
    out = build_features(_history(n_weeks=20, store=5), drop_warmup=False)
    assert out["store"].dtype.kind in ("i", "u")


def test_missing_required_column_raises():
    df = _history(n_weeks=20).drop(columns=["cpi"])
    with pytest.raises(ValueError, match="Missing required"):
        build_features(df)


def test_build_features_works_without_weekly_sales():
    """Inference-only uploads (no weekly_sales) should still produce features.

    Lag columns will be all-NaN; the caller is expected to fill them via
    `fill_warmup_with_fallback` before predicting.
    """
    df = _history(n_weeks=5).drop(columns=["weekly_sales"])
    out = build_features(df, drop_warmup=False)
    for col in LAG_COLUMNS:
        assert col in out.columns
        assert out[col].isna().all()


def test_fill_warmup_with_fallback_replaces_nan_with_per_store_means():
    df = pd.DataFrame(
        {
            "store": [1, 1, 2, 2],
            "lag_1_sales": [np.nan, 100.0, np.nan, 200.0],
            "lag_2_sales": [np.nan, np.nan, np.nan, np.nan],
            "lag_4_sales": [np.nan, np.nan, np.nan, np.nan],
            "rolling_mean_4": [np.nan, np.nan, np.nan, np.nan],
            "rolling_mean_12": [np.nan, np.nan, np.nan, np.nan],
            "rolling_std_4": [np.nan, np.nan, np.nan, np.nan],
        }
    )
    store_stats = pd.DataFrame(
        {"mean": {1: 1000.0, 2: 2000.0}, "std": {1: 50.0, 2: 75.0}}
    )
    out = fill_warmup_with_fallback(df, store_stats)

    # NaN means filled per store; existing values preserved.
    assert out.loc[0, "lag_1_sales"] == 1000.0
    assert out.loc[1, "lag_1_sales"] == 100.0
    assert out.loc[2, "lag_1_sales"] == 2000.0
    assert out.loc[3, "lag_1_sales"] == 200.0
    assert out.loc[0, "rolling_mean_12"] == 1000.0

    # rolling_std_4 uses the std fallback, not the mean.
    assert out.loc[0, "rolling_std_4"] == 50.0
    assert out.loc[2, "rolling_std_4"] == 75.0


def test_fill_warmup_with_fallback_handles_unseen_store():
    """A store absent from store_stats should fall back to dataset-wide mean/std."""
    df = pd.DataFrame(
        {
            "store": [99],
            "lag_1_sales": [np.nan],
            "lag_2_sales": [np.nan],
            "lag_4_sales": [np.nan],
            "rolling_mean_4": [np.nan],
            "rolling_mean_12": [np.nan],
            "rolling_std_4": [np.nan],
        }
    )
    store_stats = pd.DataFrame(
        {"mean": {1: 1000.0, 2: 3000.0}, "std": {1: 50.0, 2: 150.0}}
    )
    out = fill_warmup_with_fallback(df, store_stats)
    # mean of [1000, 3000] = 2000; mean of [50, 150] = 100
    assert out.loc[0, "lag_1_sales"] == 2000.0
    assert out.loc[0, "rolling_std_4"] == 100.0
