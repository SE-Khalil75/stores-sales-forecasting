"""Streamlit app for store-sales weekly forecasting."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make `src` importable when launching as `streamlit run app/streamlit_app.py`
# from the project root (Streamlit only adds the script's dir to sys.path).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import compute_metrics  # noqa: E402
from src.features import (  # noqa: E402
    LAG_COLUMNS,
    REQUIRED_INPUT_COLUMNS,
    build_features,
    fill_warmup_with_fallback,
)

MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def featurize(df: pd.DataFrame) -> pd.DataFrame:
    # drop_warmup=False so we can show the user how many rows were warm-up.
    return build_features(df, drop_warmup=False)


def _validate_columns(raw: pd.DataFrame) -> list:
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in raw.columns]
    if "fuel_Price" not in raw.columns and "fuel_price" not in raw.columns:
        missing.append("fuel_Price")
    return missing


def main() -> None:
    st.set_page_config(page_title="Store Sales Forecast", layout="wide")
    st.title("Weekly Store Sales Forecast")
    st.write(
        "Upload a CSV with the same columns as `stores-sales.csv`. "
        "`weekly_sales` is optional — if it's missing or there isn't enough "
        "per-store history to compute lag features, those values are filled "
        "with each store's training-period mean (so even a single-row upload "
        "produces a prediction). Rows that *do* have full lag context use "
        "their real history."
    )

    artifact = load_model()
    if artifact is None:
        st.error(
            f"No trained model found at `{MODEL_PATH.relative_to(PROJECT_ROOT)}`. "
            "Run `python -m src.train` first."
        )
        st.stop()

    model = artifact["model"]
    feature_cols = artifact["feature_columns"]
    lag_cols = artifact.get("lag_columns", LAG_COLUMNS)
    store_stats = artifact.get("store_stats")
    if store_stats is None:
        st.error(
            "This model artifact was trained without per-store fallback stats. "
            "Re-run `python -m src.train` to refresh `models/model.pkl`."
        )
        st.stop()

    with st.sidebar:
        st.header("Model")
        st.metric("Test RMSE", f"{artifact['test_metrics']['rmse']:,.0f}")
        st.metric("Test MAE", f"{artifact['test_metrics']['mae']:,.0f}")
        st.metric("Test MAPE", f"{artifact['test_metrics']['mape']:.2%}")
        st.caption(
            f"Trained on {artifact['n_train_rows']:,} rows "
            f"(date < {artifact['train_test_cutoff']}); "
            f"tested on {artifact['n_test_rows']:,} rows."
        )

    st.header("1. Upload data")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to begin.")
        return

    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    missing = _validate_columns(raw)
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    try:
        features_df = featurize(raw)
    except Exception as exc:
        st.error(f"Feature engineering failed: {exc}")
        return

    n_total = len(features_df)
    has_full_history = features_df[lag_cols].notna().all(axis=1)
    n_full = int(has_full_history.sum())
    n_fallback = n_total - n_full

    # Fill NaN lag/rolling values with per-store training fallbacks so we
    # can predict on every uploaded row, regardless of history length.
    features_df = fill_warmup_with_fallback(features_df, store_stats)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows uploaded", f"{n_total:,}")
    c2.metric("Stores", features_df["store"].nunique())
    c3.metric("Real / fallback lag", f"{n_full:,} / {n_fallback:,}")
    c4.metric(
        "Date range",
        f"{features_df['date'].min().date()} → {features_df['date'].max().date()}",
    )
    with st.expander("Preview uploaded data"):
        st.dataframe(raw.head(20), use_container_width=True)

    st.header("2. Predictions")
    preds = model.predict(features_df[feature_cols])
    results = features_df[["store", "date"]].copy()
    results["predicted_weekly_sales"] = preds
    results["lag_history"] = np.where(has_full_history, "real", "fallback")

    has_actual = "weekly_sales" in raw.columns
    if has_actual:
        # Align actuals to features_df (which is sorted by store, date).
        actuals = features_df["weekly_sales"]
        if actuals.notna().any():
            results["actual_weekly_sales"] = actuals.values
            mask = results["actual_weekly_sales"].notna()
            metrics = compute_metrics(
                results.loc[mask, "actual_weekly_sales"],
                results.loc[mask, "predicted_weekly_sales"],
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE", f"{metrics['rmse']:,.0f}")
            m2.metric("MAE", f"{metrics['mae']:,.0f}")
            m3.metric("MAPE", f"{metrics['mape']:.2%}")
            st.caption(
                f"Metrics computed on the {mask.sum():,} rows with non-null "
                "actual weekly_sales (predictions are produced for all rows)."
            )

    st.header("3. Results")
    display = results.sort_values(["store", "date"])
    st.dataframe(display.head(500), use_container_width=True, hide_index=True)
    if len(display) > 500:
        st.caption(
            f"Showing first 500 of {len(display):,} rows — download for the full table."
        )

    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions CSV",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv",
    )

    st.subheader("Sales over time — by store")
    selected_store = st.selectbox(
        "Store", sorted(results["store"].unique().tolist())
    )
    store_data = results[results["store"] == selected_store].sort_values("date")

    fig = go.Figure()
    if "actual_weekly_sales" in store_data.columns:
        fig.add_trace(
            go.Scatter(
                x=store_data["date"],
                y=store_data["actual_weekly_sales"],
                mode="lines+markers",
                name="Actual",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=store_data["date"],
            y=store_data["predicted_weekly_sales"],
            mode="lines+markers",
            name="Predicted",
        )
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Weekly sales",
        hovermode="x unified",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
