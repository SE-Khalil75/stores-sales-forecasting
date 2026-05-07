"""Feature engineering shared by training and the Streamlit app.

`build_features` is the single entry point. It produces:
  - Calendar parts: year, month, week_of_year, quarter, day_of_year
  - Cyclical encoding: week_sin/cos, month_sin/cos
  - Linear time trend: days_since_start
  - Holiday flags: is_super_bowl, is_labour_day, is_thanksgiving, is_christmas
  - Pre-holiday flags: same set, one week earlier
  - Lag features: lag_1_sales, lag_2_sales, lag_4_sales (per store)
  - Rolling stats: rolling_mean_4, rolling_mean_12, rolling_std_4 (per store,
    each with shift(1) so the window doesn't include the current row)

Lag and rolling features require historical `weekly_sales` per store. The
first ~12 weeks of each store's history have NaNs in some lag columns;
`drop_warmup=True` (default) removes them before returning.
"""

import numpy as np
import pandas as pd

SUPER_BOWL_DATES = pd.to_datetime(
    ["2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"]
)
LABOUR_DAY_DATES = pd.to_datetime(
    ["2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"]
)
THANKSGIVING_DATES = pd.to_datetime(
    ["2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"]
)
CHRISTMAS_DATES = pd.to_datetime(
    ["2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"]
)
_WEEK = pd.Timedelta(days=7)
PRE_SUPER_BOWL_DATES = SUPER_BOWL_DATES - _WEEK
PRE_LABOUR_DAY_DATES = LABOUR_DAY_DATES - _WEEK
PRE_THANKSGIVING_DATES = THANKSGIVING_DATES - _WEEK
PRE_CHRISTMAS_DATES = CHRISTMAS_DATES - _WEEK

LAG_COLUMNS = [
    "lag_1_sales",
    "lag_2_sales",
    "lag_4_sales",
    "rolling_mean_4",
    "rolling_mean_12",
    "rolling_std_4",
]

NUMERIC_FEATURES = [
    "holiday_flag",
    "temperature",
    "fuel_price",
    "cpi",
    "unemployment",
    "year",
    "month",
    "week_of_year",
    "quarter",
    "day_of_year",
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",
    "days_since_start",
    "is_super_bowl",
    "is_labour_day",
    "is_thanksgiving",
    "is_christmas",
    "is_pre_super_bowl",
    "is_pre_labour_day",
    "is_pre_thanksgiving",
    "is_pre_christmas",
]

FEATURE_COLUMNS = ["store"] + NUMERIC_FEATURES + LAG_COLUMNS
CATEGORICAL_FEATURES = ["store"]

REQUIRED_INPUT_COLUMNS = [
    "store",
    "date",
    "holiday_flag",
    "temperature",
    "cpi",
    "unemployment",
]
# `weekly_sales` is needed to compute lag features but not strictly required
# at inference — missing/short uploads get filled with per-store fallbacks
# via `fill_warmup_with_fallback`.


def build_features(df: pd.DataFrame, *, drop_warmup: bool = True) -> pd.DataFrame:
    """Compute calendar, holiday, and lag/rolling features.

    Required input columns: store, date, weekly_sales, holiday_flag,
    temperature, fuel_Price (or fuel_price), cpi, unemployment.

    Output is sorted by (store, date). When `drop_warmup=True` (default),
    rows with NaN in any of LAG_COLUMNS are removed; that's the first ~12
    weeks of each store's history where rolling_mean_12 isn't yet defined.
    """
    out = df.copy()

    if "fuel_Price" in out.columns:
        out = out.rename(columns={"fuel_Price": "fuel_price"})

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in out.columns]
    if "fuel_price" not in out.columns:
        missing.append("fuel_Price")
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = _add_basic_features(out)
    out = _add_lag_features(out)
    if drop_warmup:
        out = out.dropna(subset=LAG_COLUMNS).reset_index(drop=True)
    return out


def _add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Source CSV uses DD-MM-YYYY; without dayfirst, "05-02-2010" parses as May 2.
    out["date"] = pd.to_datetime(out["date"], dayfirst=True)
    out["store"] = out["store"].astype(int)

    out["year"] = out["date"].dt.year.astype(int)
    out["month"] = out["date"].dt.month.astype(int)
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)
    out["quarter"] = out["date"].dt.quarter.astype(int)
    out["day_of_year"] = out["date"].dt.dayofyear.astype(int)

    # Cyclical: smooth periodic seasonality, no jump from week 52 -> 1.
    out["week_sin"] = np.sin(2 * np.pi * out["week_of_year"] / 52)
    out["week_cos"] = np.cos(2 * np.pi * out["week_of_year"] / 52)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    out["days_since_start"] = (out["date"] - out["date"].min()).dt.days

    out["is_super_bowl"] = out["date"].isin(SUPER_BOWL_DATES).astype(int)
    out["is_labour_day"] = out["date"].isin(LABOUR_DAY_DATES).astype(int)
    out["is_thanksgiving"] = out["date"].isin(THANKSGIVING_DATES).astype(int)
    out["is_christmas"] = out["date"].isin(CHRISTMAS_DATES).astype(int)

    out["is_pre_super_bowl"] = out["date"].isin(PRE_SUPER_BOWL_DATES).astype(int)
    out["is_pre_labour_day"] = out["date"].isin(PRE_LABOUR_DAY_DATES).astype(int)
    out["is_pre_thanksgiving"] = out["date"].isin(PRE_THANKSGIVING_DATES).astype(int)
    out["is_pre_christmas"] = out["date"].isin(PRE_CHRISTMAS_DATES).astype(int)

    return out


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-store lag and rolling stats on weekly_sales.

    Sorts by (store, date) so within-group shifts/rollings are time-aligned.
    `.shift(1)` before `.rolling(N)` keeps the current row out of its own window.

    If `weekly_sales` is absent (inference-only upload), all lag columns are
    created as NaN and `fill_warmup_with_fallback` is expected to fill them.
    """
    out = df.sort_values(["store", "date"]).copy()

    if "weekly_sales" not in out.columns:
        for col in LAG_COLUMNS:
            out[col] = np.nan
        return out.reset_index(drop=True)

    g = out.groupby("store")["weekly_sales"]
    for lag in (1, 2, 4):
        out[f"lag_{lag}_sales"] = g.transform(lambda s, k=lag: s.shift(k))
    for win in (4, 12):
        out[f"rolling_mean_{win}"] = g.transform(
            lambda s, w=win: s.shift(1).rolling(w).mean()
        )
    out["rolling_std_4"] = g.transform(lambda s: s.shift(1).rolling(4).std())

    return out.reset_index(drop=True)


def fill_warmup_with_fallback(
    df: pd.DataFrame, store_stats: pd.DataFrame
) -> pd.DataFrame:
    """Replace NaN lag values with per-store fallbacks from training data.

    `store_stats` is a DataFrame indexed by store with columns `mean` and
    `std` (the per-store training mean and std of weekly_sales). Lag and
    rolling-mean NaNs are filled with the store mean; rolling_std_4 NaNs
    with the store std. Stores absent from `store_stats` (unseen at training)
    fall back to the dataset-wide mean/std.
    """
    out = df.copy()

    fallback_mean = out["store"].map(store_stats["mean"])
    fallback_mean = fallback_mean.fillna(store_stats["mean"].mean())
    fallback_std = out["store"].map(store_stats["std"])
    fallback_std = fallback_std.fillna(store_stats["std"].mean())

    for col in (
        "lag_1_sales",
        "lag_2_sales",
        "lag_4_sales",
        "rolling_mean_4",
        "rolling_mean_12",
    ):
        out[col] = out[col].fillna(fallback_mean)
    out["rolling_std_4"] = out["rolling_std_4"].fillna(fallback_std)

    return out
