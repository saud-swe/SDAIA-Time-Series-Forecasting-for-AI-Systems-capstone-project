# Technical documentation

What the notebook does, how each component works, and — more usefully — why
each design decision was made rather than its alternative. The notebook's own
markdown cells explain the *analysis*; this document explains the
*engineering and the choices*.

---

## 1. Data

| | |
|---|---|
| Source | `data/retail_demand.csv`, course repository, fetched at run time |
| Filter | `region == "Riyadh"` and `category == "Grocery"` |
| Shape | 1096 daily observations, 2023-01-01 → 2025-12-31, no gaps after `asfreq("D")` |
| Target | `units_sold`, float, min 397, median 626, mean 669.5, max 2357, **no zeros** |
| Provenance | synthetic, `data/generate_series.py`, `SEED = 20260912`, deterministic |

The generating process (documented in the course's generator) composes a
multiplicative model: a level per region/category, a linear trend, a 7-day
weekday multiplier, a yearly sinusoid, drifting Hijri-like holiday bumps, a
handful of 2–4 day promo shocks per year, lognormal-ish noise, and a final
Poisson draw. Several findings in the notebook are only interpretable against
that structure, which is why it is stated rather than reverse-engineered.

No data is vendored into this repository. The notebook's `fetch()` helper —
copied verbatim from the course labs — looks for a local checkout first and
falls back to `raw.githubusercontent.com`, so the same file runs from a clone,
from Colab, and inside the course's `colab-sim` container.

## 2. Transformation

All classical modelling is done on `log(units_sold)`.

**Why.** STL residual standard deviation on the raw scale grows 18% between
the first and second halves of the sample; on the log scale it grows 3%. That
is the operational definition of multiplicative seasonality. Seasonality
strength also rises (0.467 → 0.626) on the log scale, i.e. the weekly
component is cleaner once the level effect is divided out rather than
subtracted.

Forecasts are exponentiated back to the original units before scoring, so
every metric in the notebook is in `units_sold`, comparable across models
regardless of the scale each was fitted on.

LightGBM is fitted on the **raw** scale. Trees are invariant to monotone
transforms of the target only in their split structure, not in their leaf
values, but the practical reason is simpler: the log transform exists to
stabilise variance for a model that assumes constant variance, and a GBM makes
no such assumption.

## 3. Models

### Seasonal-naive (baseline)

`common/backtest.py`'s `seasonal_naive_forecast` with `period=7`. Present in
every comparison because a model that cannot beat "repeat last week" should
not be deployed, and because MASE is scaled against exactly this forecast —
so it also fixes the interpretation of every MASE figure in the report.

### SARIMA(1,0,1)(1,1,1)[7] with constant, on log

Selected in two stages, both shown in the notebook.

*Stage 1 — a candidate grid from the ACF/PACF read, not a blind sweep.* Seven
specifications, scored by AIC and BIC. Every `d = 1` candidate lost, confirming
that a first difference on top of the seasonal difference over-differences.

*Stage 2 — residual adequacy as a tie-breaker.* The top three AIC values span
1.6 points, which conventional practice treats as indistinguishable. BIC
separates them and prefers the 5-parameter `(1,0,1)(0,1,1)[7]`. But that model
leaves **significant Ljung-Box autocorrelation at lag 7** (p = 0.046) — the
weekly lag, the dominant structure in the series. Adding the seasonal AR term
back raises that to p = 0.41, and the term's own coefficient is significant
(p = 0.040).

**Decision: the seasonal AR term stays**, against BIC's ~6-point preference.
An information criterion scores in-sample likelihood against parameter count;
it says nothing about whether the residuals still carry structure. Leftover
autocorrelation at the seasonal lag means systematic mis-forecasting of the
same weekday, which propagates directly into both the backtest and the
parametric intervals. The diagnostic outranks the criterion here.

`enforce_stationarity=False, enforce_invertibility=False` because the seasonal
MA coefficient sits at the invertibility boundary (≈ −1.01); constraining it
degrades the fit without changing the forecasts materially.

### Holt-Winters (additive trend, multiplicative seasonal, m=7)

A second classical family, included as a check that the SARIMA result is not
an artefact of one model class. Its fitted parameters are informative in
themselves: `beta` and `gamma` both optimise to ≈ 0, meaning the trend and the
weekly shape are stable enough that the model never updates either after
initialisation. That is consistent with the smooth STL trend — and it is also
the model's limitation, since a frozen seasonal index cannot respond to
holiday bumps at all.

### LightGBM + recursive multi-step forecasting

Twelve features in four groups:

| Group | Features |
|---|---|
| Autoregressive | `lag_1`, `lag_7`, `lag_14`, `lag_28` |
| Local level and volatility | `roll_mean_7`, `roll_std_7`, `roll_mean_28` |
| Calendar | `dow`, `is_weekend` (Fri/Sat, KSA weekend), `month` |
| Smooth yearly | `doy_sin`, `doy_cos` |

Fixed hyper-parameters (`n_estimators=400`, `learning_rate=0.05`,
`max_depth=6`, `num_leaves=31`, `subsample=0.9`, `colsample_bytree=0.9`,
`random_state=RNG_SEED`). **Not tuned** — see Limitations.

Multi-step forecasts are produced **recursively**: each step appends its own
prediction to the history and recomputes features from the extended history,
so a lag pointing into the forecast window resolves to a model prediction, not
a true future value. The notebook logs how many of the four lag features are
synthetic at each step (0 at step 1, rising to 2 by step 8), which is the
mechanical explanation for why error grows with horizon.

## 4. Leakage control

The single most expensive class of error in applied forecasting, so it is
handled structurally and then *verified*, not asserted.

**Structural guarantees**

1. Every lag and rolling statistic derives from `series.shift(1)` or further
   back. The window for date *t* closes at *t−1*. Calendar features are exempt
   because a future date's day-of-week is known in advance.
2. `make_features` is called **inside each backtest fold**, on that fold's
   training data only — never once on the full series. No global statistic,
   scaler or imputation is fitted anywhere.
3. Recursive forecasting feeds the model its own predictions, never held-out
   actuals.

**Runtime assertions** (these fail the notebook loudly rather than producing a
quietly wrong number)

- A mechanical audit re-derives `lag_1`, `lag_7`, `roll_mean_7` and
  `roll_mean_28` for sampled dates from strictly-past data and asserts each
  matches, and that no feature equals the target it predicts.
- The recursive loop asserts, at every step, that the forecast date is absent
  from history and that history ends exactly one day before it.
- The fold generator is checked for train/test overlap, contiguity of test
  windows, monotone expansion of train windows, correct horizon, and that the
  final fold ends at the last observation in the file.

## 5. Backtesting

`common/backtest.py`'s `expanding_window_splits` + `run_backtest`:
**8 folds, 14-day horizon, 730-day minimum training window**, covering
2025-09-11 → 2025-12-31 (112 held-out days). Every model is refit from scratch
on every fold.

**Expanding, not rolling.** The STL trend is smooth and monotone with no level
break, so a day from early 2023 is generated by the same process (up to a
level shift the trend term handles) as a day in late 2025. When old history is
still informative, discarding it costs estimation precision and buys nothing —
and SARIMA's seasonal terms in particular want as many complete weekly cycles
as they can get. The contrast case is `workforce_demand.csv`, whose structural
break on 2025-04-01 makes a rolling window the correct design there.

That argument is then *tested* rather than left as an assertion: the notebook
re-runs the full comparison under a fixed 730-day rolling window. The ranking
is unchanged (SARIMA improves modestly; LightGBM is flat), which is a null
result and is reported as one.

**Why 8 folds and not 3.** Three is the brief's minimum. The notebook shows
that on this series a 3-fold backtest agrees with the single holdout and
therefore picks the *wrong* model; the verdict only flips at higher fold
counts. Three folds is the floor, not a safe number.

**Date alignment inside `run_backtest`.** The harness passes bare numpy arrays,
but the LightGBM path needs a `DatetimeIndex` for its calendar features. It
reconstructs one from the front of the series using `len(y_train)`, which is
valid because expanding windows always start at index 0 (asserted), and
because a future date's calendar is known in advance regardless.

## 6. Metrics

| Metric | Role |
|---|---|
| **MASE** (sp=7) | Primary. Scaled against the in-sample seasonal-naive MAE, so < 1.0 means "beats repeating last week" and the number is comparable across series. |
| **WAPE** | Secondary scale-free. Total absolute error over total actual volume — reads as "off by x% of volume moved". |
| MAE | Reported, in series units. Concrete but not comparable across series. |
| RMSE | Reported alongside MAE; the gap between them measures how much the promo shocks dominate. |
| MAPE | Reported, **not ranked on**. See below. |

**On MAPE.** The brief requires WAPE over MAPE where a series has zeros or
near-zeros, and requires the reason to be stated. This series has **no zeros**
(minimum 397), so MAPE is well-defined and is reported rather than suppressed.
It is still not the ranking metric, for a reason that survives the absence of
zeros: MAPE divides each error by that day's own actual, so it penalises
over-forecasting a quiet Tuesday more harshly than under-forecasting a promo
Saturday by the same number of units. On a series whose difficulty is
sporadic upward spikes, that asymmetry quietly rewards a model that forecasts
systematically low. WAPE weights by volume and has no such preference.

Every metric is computed **per fold** as well as pooled, because the
fold-to-fold spread turns out to be the same size as the gaps between models —
which is the single most decision-relevant fact the backtest produces.

## 7. Probabilistic forecasting

Two 80% intervals, scored against each other.

**(a) Split-conformal on LightGBM.** Within each fold's training data, the
final 6 × 14 = 84 days are held out as a calibration block. The model is fit on
everything before it, 14-day forecasts are rolled across the calibration
block, and absolute residuals are collected per horizon step. Half-widths are
the finite-sample conformal quantile `ceil((n+1)(1−α))/n` taken separately for
steps 1–7 and steps 8–14. Distribution-free: it assumes only that calibration
residuals are exchangeable with test residuals, never that they are Gaussian.

**(b) SARIMA's native parametric interval**, from `get_forecast().conf_int()`,
exponentiated back from log.

**Result.** Pooled over 112 held-out points:

| | empirical coverage | mean width | width as % of mean demand |
|---|--:|--:|--:|
| Conformal LightGBM | 87.5% | 147.0 | 25.8% |
| SARIMA native | 98.2% | 278.2 | 48.8% |

SARIMA wins on coverage and loses badly on the pair. Its interval spans
roughly half of typical daily volume, which is close to useless for the
stocking decision it exists to support. The cause is identified rather than
guessed: the parametric interval assumes Gaussian, serially uncorrelated
residuals, and Section 2's Ljung-Box test and Q-Q plot show the residuals are
fat-tailed and clustered because of promo shocks. Estimating a Gaussian
variance from a fat-tailed sample inflates it at every horizon step.

Two caveats are reported rather than smoothed over: the conformal interval is
**conservative** (87.5% against a nominal 80%), and its step-bucket half-widths
came out **non-monotone** in horizon — within sampling noise at 6 calibration
blocks per bucket, but stated rather than presented as a clean result.

## 8. Reproducibility and verification

- `RNG_SEED = 20260912` seeds numpy and LightGBM.
- `build/build_capstone.py` generates the notebook; `build/execute.py`
  executes it in place and then verifies: every code cell has a non-null
  `execution_count`, no cell produced an error output, no markdown cell
  `source` is a flat string, and no markdown cell begins with a bare `---`
  (the last two are the Quarto notebook traps documented in the course
  repository's own `CLAUDE.md`).
- Full run: ~65 s on a 2-core container; ~90 s on a cold Colab runtime
  including installs.
- Environment used for the committed run: Python 3.11, pandas 3.0.2,
  numpy 2.4.4, statsmodels 0.15.0, LightGBM 4.7.0, scikit-learn 1.8.0.

## 9. Limitations

1. **Synthetic data.** Every conclusion describes `generate_series.py`'s
   output. Real grocery demand has stockouts, price effects, competitor
   actions and reporting errors, none of which are present.
2. **SARIMA residuals are not white noise.** Ljung-Box rejects at lags 14–28
   even for the chosen specification. The diagnosis is a *missing regressor*
   (the drifting holiday calendar), not a misspecified ARMA order, so the
   model was not patched with extra terms to force the test to pass.
3. **The backtest covers 112 days**, all in the second half of 2025. It does
   not test behaviour across the April holiday peak. Extending it is the first
   thing I would do with more time.
4. **The conformal interval is conservative** and its half-widths are
   non-monotone in horizon.
5. **One series of six.** The recommendation to go global across all six
   region × category pairs is an argument from structure, not a demonstrated
   result.
6. **No hyper-parameter search.** LightGBM uses fixed, reasonable parameters.
   Tuning inside each fold is the correct approach and costs roughly an order
   of magnitude more compute; tuning once on the full series would be leakage,
   which is why neither was done.
