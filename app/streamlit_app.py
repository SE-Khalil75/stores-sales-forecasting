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

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_VARIANTS = {
    "With lag features": MODELS_DIR / "model.pkl",
    "Without lag features": MODELS_DIR / "model_no_lag.pkl",
}


@st.cache_resource
def load_model(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return None
    return joblib.load(path)


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
    st.caption(
        "Upload a CSV with the same columns as `stores-sales.csv` "
        "to generate weekly sales predictions."
    )

    available = {
        label: path for label, path in MODEL_VARIANTS.items() if path.exists()
    }
    if not available:
        st.error(
            "No trained model found in `models/`. "
            "Run `python -m src.train` (and optionally `--no-lag`) first."
        )
        st.stop()

    with st.sidebar:
        st.subheader("Model")
        choice = st.radio(
            "Variant",
            list(available.keys()),
            label_visibility="collapsed",
            help=(
                "‘With lag features’ uses per-store lag/rolling stats. "
                "‘Without’ uses only calendar, holiday, and exogenous features."
            ),
        )

    artifact = load_model(str(available[choice]))
    model = artifact["model"]
    feature_cols = artifact["feature_columns"]
    uses_lag = artifact.get("uses_lag_features", "store_stats" in artifact)
    lag_cols = artifact.get("lag_columns", LAG_COLUMNS)
    store_stats = artifact.get("store_stats")

    with st.sidebar:
        st.divider()
        st.subheader("Test performance")
        st.metric("RMSE", f"{artifact['test_metrics']['rmse']:,.0f}")
        st.metric("MAE", f"{artifact['test_metrics']['mae']:,.0f}")
        st.metric("MAPE", f"{artifact['test_metrics']['mape']:.2%}")
        st.caption(
            f"Trained on {artifact['n_train_rows']:,} rows "
            f"(date < {artifact['train_test_cutoff']}); "
            f"tested on {artifact['n_test_rows']:,} rows."
        )

    st.divider()
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to begin.")
        with st.expander("How predictions are produced"):
            st.markdown(
                "- `weekly_sales` is **optional**. If a row lacks enough "
                "per-store history to compute lag features (or the column "
                "is missing), values are filled with each store's "
                "training-period mean — so even a single-row upload predicts.\n"
                "- Rows with full lag context use their **real** history.\n"
                "- The **Without lag features** model skips lag/rolling "
                "entirely and depends only on calendar, holiday, and "
                "exogenous variables (temperature, fuel, CPI, unemployment)."
            )
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

    with st.expander(f"Preview ({len(raw):,} rows)"):
        st.dataframe(raw.head(5), use_container_width=True, hide_index=True)

    try:
        features_df = featurize(raw)
    except Exception as exc:
        st.error(f"Feature engineering failed: {exc}")
        return

    n_total = len(features_df)
    if uses_lag:
        has_full_history = features_df[lag_cols].notna().all(axis=1)
        n_full = int(has_full_history.sum())
        n_fallback = n_total - n_full
        # Fill NaN lag/rolling values with per-store training fallbacks so we
        # can predict on every uploaded row, regardless of history length.
        features_df = fill_warmup_with_fallback(features_df, store_stats)
    else:
        has_full_history = pd.Series(True, index=features_df.index)
        n_full, n_fallback = n_total, 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{n_total:,}")
    c2.metric("Stores", features_df["store"].nunique())
    if uses_lag:
        c3.metric("Real / fallback lag", f"{n_full:,} / {n_fallback:,}")
    else:
        c3.metric("Lag features", "disabled")
    c4.metric(
        "Date range",
        f"{features_df['date'].min().date()} → {features_df['date'].max().date()}",
    )

    preds = model.predict(features_df[feature_cols])
    results = features_df[["store", "date"]].copy()
    results["predicted_weekly_sales"] = preds
    if uses_lag:
        results["lag_history"] = np.where(has_full_history, "real", "fallback")

    has_actual = "weekly_sales" in raw.columns
    actuals_aligned = features_df["weekly_sales"] if has_actual else None
    if has_actual and actuals_aligned.notna().any():
        results["actual_weekly_sales"] = actuals_aligned.values
        mask = results["actual_weekly_sales"].notna()
        metrics = compute_metrics(
            results.loc[mask, "actual_weekly_sales"],
            results.loc[mask, "predicted_weekly_sales"],
        )
        st.divider()
        st.subheader("Accuracy on this upload")
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", f"{metrics['rmse']:,.0f}")
        m2.metric("MAE", f"{metrics['mae']:,.0f}")
        m3.metric("MAPE", f"{metrics['mape']:.2%}")
        st.caption(
            f"Computed on the {mask.sum():,} rows with non-null "
            "actual weekly_sales."
        )

    st.divider()
    table_tab, store_tab = st.tabs(["Predictions table", "By store"])

    with table_tab:
        display = results.sort_values(["store", "date"])
        st.dataframe(display.head(500), use_container_width=True, hide_index=True)
        if len(display) > 500:
            st.caption(
                f"Showing first 500 of {len(display):,} rows — "
                "download for the full table."
            )
        st.download_button(
            "Download predictions CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

    with store_tab:
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
                    line=dict(color="#1f77b4", width=2),
                    marker=dict(size=6),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=store_data["date"],
                y=store_data["predicted_weekly_sales"],
                mode="lines+markers",
                name="Predicted",
                line=dict(color="#ff7f0e", width=2, dash="dot"),
                marker=dict(size=6),
            )
        )
        fig.update_layout(
            title=f"Store {selected_store} — weekly sales",
            xaxis_title=None,
            yaxis_title="Weekly sales",
            hovermode="x unified",
            height=440,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
