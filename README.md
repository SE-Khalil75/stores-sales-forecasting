# Store Sales Forecasting

Weekly sales forecasting for 45 retail stores. A LightGBM regressor with
lag, rolling, cyclical, and holiday features is trained on historical
weekly sales (Feb 2010 – Oct 2012) and served through a Streamlit app
that takes a CSV upload and returns predicted weekly sales per store.

## Demo

[![Streamlit app demo](https://cdn.loom.com/sessions/thumbnails/2374715778b549bfab8523f8e4fe7421-with-play.gif)](https://www.loom.com/share/2374715778b549bfab8523f8e4fe7421)

▶ [Watch the Streamlit app demo on Loom](https://www.loom.com/share/2374715778b549bfab8523f8e4fe7421)

## Reported metrics

Trained on 4,905 rows after the warm-up drop (`date < 2012-06-01`); held-out
test of 990 rows (`date >= 2012-06-01`).

| Split | RMSE     | MAE      | MAPE  |
| ----- | -------- | -------- | ----- |
| Train | 39,956.1 | 25,784.4 | 2.78% |
| Test  | 53,378.0 | 35,953.1 | 3.51% |

For context, predicting each store's own historical mean every week gives a
test RMSE of ~$96k. The model is ~45% better than that baseline and ~10×
better than predicting the global mean.

Top feature importances: `lag_1_sales`, `week_of_year`, `lag_4_sales`,
`rolling_std_4`, `week_cos`, `temperature`, `day_of_year`, `rolling_mean_4`,
`rolling_mean_12`, `lag_2_sales`. The lag and rolling features dominate.

## Setup

```bash
git clone <repo>
cd store-sales-forecasting
bash setup.sh                       # creates venv/, installs requirements.txt
source venv/Scripts/activate        # Git Bash on Windows
# source venv/bin/activate          # macOS / Linux
```

Place the dataset at `data/stores-sales.csv` (the data folder is gitignored
except for `.gitkeep`).

Tested on Python 3.9.6 and 3.11+. Requires the package versions pinned in
[requirements.txt](requirements.txt) (LightGBM, XGBoost, scikit-learn,
Streamlit, Plotly, Jupyter).

## Train the model

Two equivalent paths — both produce the same `models/model.pkl` consumed by
the Streamlit app.

**CLI** (the production path):

```bash
python -m src.train
```

**Notebook** (the exploration path):

```bash
jupyter lab notebooks/01_train_and_evaluate.ipynb
# or: jupyter notebook notebooks/01_train_and_evaluate.ipynb
```

The notebook is self-contained (defines all feature engineering inline) and
also compares LightGBM against XGBoost, Random Forest, and Ridge with a
simple ensemble at the end. The CLI is the production refactor of that
prototype using the shared code under `src/`.

## Run the app

```bash
streamlit run app/streamlit_app.py
```

The app:

1. **Upload data** — accepts a CSV with the same columns as `stores-sales.csv`.
   `weekly_sales` is **optional**: if it's present and the store has enough
   history, the lag and rolling features are computed normally; if it's
   missing or short, those values are filled with that store's
   training-period mean (so even a single-row upload returns a prediction).
   Required columns: `store`, `date`, `holiday_flag`, `temperature`,
   `fuel_Price` (or `fuel_price`), `cpi`, `unemployment`.
2. **Predictions** — applies the same `build_features` pipeline as training,
   fills any NaN lag/rolling values via `fill_warmup_with_fallback` using
   per-store training stats baked into the model artifact, then predicts on
   every uploaded row. The metric cards show how many rows used real lag
   history vs. fallback. RMSE / MAE / MAPE are computed against `weekly_sales`
   when it's present.
3. **Results** — a `lag_history` column flags each row as `real` or
   `fallback`. Sortable table, a per-store Plotly line chart of predicted
   (and actual, when available) weekly sales over time, and a CSV download
   button.

## Approach

**Feature engineering.** [src/features.py](src/features.py) is the single
source of truth. It parses `DD-MM-YYYY` dates with `dayfirst=True` and
produces:

- **Calendar parts**: `year`, `month`, `week_of_year`, `quarter`, `day_of_year`.
- **Cyclical encoding**: `week_sin`/`week_cos` and `month_sin`/`month_cos`
  so seasonality is a smooth periodic signal (week 52 sits next to week 1
  on the unit circle).
- **Linear time trend**: `days_since_start` for slow secular drift.
- **Holiday flags**: `is_super_bowl`, `is_labour_day`, `is_thanksgiving`,
  `is_christmas`.
- **Pre-holiday flags**: same set, one week earlier (build-up week often
  spikes too).
- **Lag features**: `lag_1_sales`, `lag_2_sales`, `lag_4_sales` per store.
- **Rolling stats**: `rolling_mean_4`, `rolling_mean_12`, `rolling_std_4`,
  each with `shift(1)` so the window doesn't include the current row.

`fuel_Price` is normalized to `fuel_price`. `store` is kept as an integer
and passed to LightGBM as a categorical feature.

The lag/rolling features imply a **1-step-ahead** forecasting setup at
training time. For inference, the model artifact also stores per-store
mean and std of `weekly_sales` from training; the app uses those as
fallbacks via `fill_warmup_with_fallback` so the user can upload arbitrary
CSVs (including a single row, or rows without `weekly_sales`) and still
get predictions. Real lag values are used whenever the upload provides
enough history; fallbacks fill the rest.

**Model.** LightGBM (`n_estimators=2000`, `learning_rate=0.03`,
`num_leaves=63`, `min_child_samples=20`, `feature_fraction=0.9`,
`bagging_fraction=0.9`, `bagging_freq=5`, `random_state=42`). Early stopping
with `stopping_rounds=50` on the inner val set guards against overfitting.

**Why LightGBM.** In a head-to-head against XGBoost, Random Forest, and
Ridge regression on the same feature set (see the notebook), LightGBM was
the best single model and an equal-weight ensemble of the top 3 actually
came in slightly worse — averaging weaker models pulls predictions toward
their errors. So the saved artifact is the single LightGBM.

**Validation strategy.** Strictly time-based — never random.

- **Train**: rows where `date < 2012-06-01` (4,905 rows after warm-up drop).
- **Test**: rows where `date >= 2012-06-01` (990 rows, ~22 weeks per store).
- **Inner val**: `2012-04-01` → `2012-05-31` (~360 rows) used solely for
  early stopping. Train metrics are reported on the full train period.

## Project structure

```
store-sales-forecasting/
├── app/
│   └── streamlit_app.py        # Upload → predict → display + download
├── data/
│   └── stores-sales.csv        # gitignored (place here)
├── models/
│   └── model.pkl               # gitignored (created by training)
├── notebooks/
│   └── 01_train_and_evaluate.ipynb   # 4-model comparison + ensemble
├── src/
│   ├── __init__.py
│   ├── features.py             # build_features, FEATURE_COLUMNS, LAG_COLUMNS
│   ├── train.py                # CLI: python -m src.train
│   └── evaluate.py             # time_based_split, compute_metrics
├── tests/
│   └── test_features.py        # 15 tests
├── conftest.py                 # adds project root to sys.path for pytest
├── requirements.txt
├── setup.sh
└── README.md
```

## Tests

```bash
pytest tests/
```

17 tests cover date parsing (`dayfirst=True`), each holiday flag, lag
correctness (and per-store isolation), the warm-up-drop behaviour, the
cyclical-encoding unit-circle invariant, all `FEATURE_COLUMNS` present, no
NaNs after warm-up drop, the `fuel_Price` rename (and idempotency when
already lowercase), `store` dtype, the required-column validation error,
`build_features` working without `weekly_sales`, and
`fill_warmup_with_fallback` filling per-store means/stds (with dataset-wide
fallback for unseen stores).

## Using the app

1. Train the model: `python -m src.train` (or run the notebook).
2. Launch: `streamlit run app/streamlit_app.py`.
3. The sidebar shows the model's reported train/test metrics.
4. Upload a CSV with the required columns above (must include
   `weekly_sales`). The upload widget will tell you how many rows were used
   for prediction vs. dropped as warm-up.
5. Browse the results table, pick a store from the dropdown to see its
   forecast over time, and download the predictions CSV.
