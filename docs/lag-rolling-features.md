# Lag and Rolling Features — Engineering Note

## What they are

Six columns added to every `(store, date)` row, all computed **per store** so
each store's lag/rolling values only ever look at its own history:

| Column            | Definition                                                |
| ----------------- | --------------------------------------------------------- |
| `lag_1_sales`     | Store's `weekly_sales` from 1 week earlier                |
| `lag_2_sales`     | Store's `weekly_sales` from 2 weeks earlier               |
| `lag_4_sales`     | Store's `weekly_sales` from 4 weeks earlier               |
| `rolling_mean_4`  | Mean of the store's previous 4 weeks of `weekly_sales`    |
| `rolling_mean_12` | Mean of the store's previous 12 weeks of `weekly_sales`   |
| `rolling_std_4`   | Std-dev of the store's previous 4 weeks of `weekly_sales` |

## Why they belong in the model

Weekly retail sales are **autoregressive**: this week's number is mostly a
function of last week's. Foot traffic, promotions, SKU mix, local employment
all change slowly. A store doing $1.5M last week is overwhelmingly more
likely to do $1.4–1.6M next week than $300k or $3M.

Without lag features the model reconstructs that level indirectly from
`store` + calendar + macro features. With `lag_1_sales`, the model gets a
near-direct readout of "where this store currently is" and only has to
predict the small delta. `rolling_mean_4`/`_12` smooth the noisy single-week
observation; `rolling_std_4` tells the model how volatile the store has been
recently (so noisy stores get less confident predictions).

## How they're computed — and how leakage is prevented

```python
out = df.sort_values(["store", "date"]).copy()
g = out.groupby("store")["weekly_sales"]

for lag in (1, 2, 4):
    out[f"lag_{lag}_sales"] = g.transform(lambda s, k=lag: s.shift(k))

for win in (4, 12):
    out[f"rolling_mean_{win}"] = g.transform(
        lambda s, w=win: s.shift(1).rolling(w).mean()
    )
out["rolling_std_4"] = g.transform(lambda s: s.shift(1).rolling(4).std())
```

Two correctness checks baked into this code:

1. **Sort by `(store, date)` before computing.** `groupby + shift` is only
   meaningful if rows are chronological within each group.
2. **`.shift(1)` before `.rolling(N)`.** Without it, the rolling window
   includes the current row's `weekly_sales` — the model would be predicting
   the target from a quantity that already contains it. The shift moves the
   window strictly into the past.

## Empirical impact

Held-out test set (4,905 train rows / 990 test rows), same LightGBM
hyperparameters in both arms, only the feature set differs:

| Metric | Basic features only | + lag / rolling | Improvement |
| ------ | ------------------: | --------------: | ----------: |
| RMSE   |              92,847 |      **53,378** | **−42.5 %** |
| MAE    |              63,973 |      **35,953** | **−43.8 %** |
| MAPE   |               7.42% |       **3.51%** | **−52.6 %** |

`lag_1_sales` is the model's #1 feature by importance; five of the top ten
features overall come from this block.

## Trade-offs

- **Warm-up cost.** `rolling_mean_12` needs 12 weeks of history per store, so
  the first 12 weeks of each store (~540 rows out of 6,435) are dropped from
  training. The lift more than compensates.
- **1-step-ahead deployment shape.** The model assumes last week's actuals
  are available when predicting next week — the standard form for
  autoregressive forecasters.
- **Inference flexibility.** Per-store mean and std of `weekly_sales` from
  the training period are saved into `models/model.pkl`. At inference,
  `fill_warmup_with_fallback` imputes NaN lag values with those stats so any
  upload (including a single row, or one without `weekly_sales`) still gets
  a prediction. The Streamlit app tags each row's `lag_history` as `real`
  or `fallback`, so consumers know which predictions to trust most.

---

## Q&A — challenges I worked through during this engineering decision

### Q. Why bother adding lag features instead of using the raw calendar/macro features alone?

Trees on calendar + store ID can recover store-level baselines, but only
through dozens of splits per prediction. Lag features hand the answer
directly to the model. The 42.5% RMSE drop in the ablation above is the
empirical answer.

### Q. If a user uploads a CSV without `weekly_sales` at all, every lag value falls back to the store's training mean. Are predictions then "just the store mean"?

Not quite. The model still has 24 other features doing real work
(calendar, holiday flags, cyclical encoding, macro indicators). But the six
lag/rolling columns collapse to constants per store, which is largely
redundant with the `store` categorical feature itself. Empirically, no-history
predictions land near the basic-features-only ceiling — RMSE around $93k,
not the $53k of the full model. Honest fallback, but flagged as `lag_history
= fallback` so downstream users don't over-trust those rows. Good enough for
"don't crash the app" use cases; for real no-history forecasting, train a
dedicated basic-features model and route to it instead.

### Q. We can always recursively predict our way to a far-future week — predict 2013, then 2014, etc., feeding each prediction back as the lag for the next. So we never really lack history, right?

Mechanically yes — recursive multi-step forecasting is a real technique.
But it has practical limits:

- **Errors compound.** Each prediction carries some RMSE; that error becomes
  an *input* to the next prediction's lag features. After a handful of
  weeks the lag inputs are entirely model-generated, and the system
  converges toward whatever level the model thinks the store should be at,
  not toward reality.
- **Macro features go out of distribution.** 2020 CPI / fuel / unemployment
  values weren't in the 2010–2012 training data. Trees extrapolate poorly
  outside the range they saw.

So: works well for short horizons (next 4–8 weeks), unreliable for long
horizons. For 2020-from-2012 specifically, the right move is to retrain with
2018–2019 data, not to recurse 8 years forward.
