# Results

Every figure this project reports, in one place, so a reader can check the
notebook's prose against its numbers without re-running anything. All of it is
the output of a single top-to-bottom run of
`capstone_forecasting_report.ipynb`.

**Setup:** Riyadh / Grocery, `data/retail_demand.csv`, 1096 daily
observations (2023-01-01 → 2025-12-31). 8-fold expanding walk-forward,
14-day horizon, 730-day minimum training window. Backtest window
2025-09-11 → 2025-12-31 (112 held-out days).

---

## Series characteristics

| | |
|---|--:|
| Observations | 1096 |
| Missing after `asfreq("D")` | 0 |
| Zero-valued days | 0 |
| Min / median / mean / max | 397 / 626 / 669.5 / 2357 |
| Days above 2× median | 22 |
| Growth 2023 → 2025 | +11.9% (≈ 6%/yr) |

Day-of-week multipliers (ratio to overall daily mean):

| Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|--:|--:|--:|--:|--:|--:|--:|
| 0.905 | 0.876 | 0.885 | 0.912 | 0.979 | **1.261** | **1.182** |

## Decomposition (STL, period 7)

| Scale | Seasonality strength | Trend strength | Residual sd, 1st half → 2nd half |
|---|--:|--:|--:|
| Raw units (additive) | 0.467 | 0.668 | 95.6 → 113.1 (×1.183) |
| log units (multiplicative) | **0.626** | **0.745** | 0.107 → 0.111 (×1.033) |

→ **multiplicative**; classical models are fitted on `log(units_sold)`.

## Stationarity

| Transform | ADF stat | ADF p | ADF verdict | KPSS p | KPSS verdict |
|---|--:|--:|---|--:|---|
| units_sold (raw) | −3.872 | 0.0023 | stationary | 0.10 | stationary |
| log(units_sold) | −3.081 | 0.0280 | stationary | 0.10 | stationary |
| log, diff(1) | −9.752 | 0.0000 | stationary | 0.10 | stationary |
| log, diff(7) | −8.347 | 0.0000 | stationary | 0.10 | stationary |
| log, diff(7) then diff(1) | −13.192 | 0.0000 | stationary | 0.10 | stationary |

**Both tests call the undifferenced series stationary.** Neither has any
machinery for a 7-day cycle. Seasonal differencing is justified by the ACF
collapsing, not by a p-value. (KPSS p = 0.10 is the ceiling of the test's
lookup table, i.e. "not close to rejecting".)

## Autocorrelation

ACF of `log(units_sold)`: lag 1 = +0.734, **lag 7 = +0.742**, lag 14 = +0.610,
lag 21 = +0.602, lag 28 = +0.589, lag 35 = +0.574 — local maxima at every
multiple of 7, no decay to zero.

After `diff(7)`: lag 1 = +0.60, decaying inside the ±0.06 band by ≈ lag 6, with
a clear negative spike at **lag 7 = −0.25** (seasonal MA fingerprint). A mild
negative band at lags 12–16 persists.

## SARIMA order selection

| # | order | seasonal_order | k | AIC | BIC |
|--:|---|---|--:|--:|--:|
| 1 | (1,0,1) | (0,1,1,7) | 5 | **−1408.13** | **−1383.75** |
| 2 | (1,0,2) | (1,1,1,7) | 7 | −1407.97 | −1373.85 |
| 3 | (1,0,1) | (1,1,1,7) | 6 | −1406.53 | −1377.28 |
| 4 | (2,1,2) | (1,1,1,7) | 7 | −1405.53 | −1371.42 |
| 5 | (1,1,1) | (1,1,1,7) | 5 | −1399.26 | −1374.89 |
| 6 | (2,0,1) | (1,1,1,7) | 7 | −1394.27 | −1360.15 |
| 7 | (0,1,1) | (0,1,1,7) | 3 | −1379.11 | −1364.48 |

Top three AIC span 1.6 points — indistinguishable. Finalists compared on
residual adequacy:

| | (1,0,1)(0,1,1)[7] | (1,0,1)(1,1,1)[7] |
|---|--:|--:|
| k | 5 | 6 |
| AIC | −1408.13 | −1406.53 |
| BIC | −1383.75 | −1377.28 |
| **Ljung-Box p @ lag 7** | **0.046** ← structure left | **0.408** ← clean |
| Ljung-Box p @ lag 14 | 0.001 | 0.006 |
| Ljung-Box p @ lag 28 | 0.006 | 0.022 |
| seasonal AR p-value | — | 0.040 |

**Chosen: SARIMA(1,0,1)(1,1,1)[7] with constant** — the diagnostic outranks
the ~6-point BIC preference.

Fitted coefficients: `ar.L1 = 0.916` (p<0.001), `ma.L1 = −0.285` (p<0.001),
`ar.S.L7 = 0.069` (p=0.040), `ma.S.L7 = −1.012` (p<0.001), `sigma2 = 0.0129`.
Residual mean −0.0008, sd 0.116, 1.96% beyond 3 sd.

## Holt-Winters

`alpha = 0.7704`, `beta = 0.0000`, `gamma = 0.0000`, SSE = 10,084,805. Trend
and seasonal shape are never updated after initialisation.

## LightGBM feature importance (% of splits)

| Feature | % | Group |
|---|--:|---|
| lag_1 | 17.3 | recent level |
| roll_mean_7 | 13.1 | recent level |
| doy_sin | 12.7 | yearly |
| roll_std_7 | 9.0 | volatility |
| roll_mean_28 | 8.9 | recent level |
| lag_28 | 8.9 | weekly |
| lag_7 | 8.2 | weekly |
| lag_14 | 8.0 | weekly |
| doy_cos | 7.2 | yearly |
| dow | 4.5 | weekly |
| month | 1.6 | calendar |
| is_weekend | 0.8 | weekly |

Groups: recent level ≈ 39%, weekly ≈ 30%, yearly ≈ 20%, volatility ≈ 9%.

Recursive lag lineage: 0 of 4 lags synthetic at step 1, 1 of 4 from step 2,
**2 of 4 from step 8**.

## The single-holdout trap

Single 14-day holdout, 2025-12-18 → 2025-12-31:

| Model | MAE | RMSE | MASE |
|---|--:|--:|--:|
| **SARIMA** | 25.36 | 29.43 | **0.286** |
| Holt-Winters | 33.45 | 39.00 | 0.377 |
| LightGBM | 34.28 | 39.97 | 0.387 |
| Seasonal-naive | 61.21 | 69.44 | 0.690 |

→ winner: **SARIMA**.

## 8-fold walk-forward — the actual result

| Model | MAE | MAE sd | RMSE | WAPE % | MASE | MASE sd |
|---|--:|--:|--:|--:|--:|--:|
| Seasonal-naive | 45.15 | 10.48 | 53.53 | 7.87 | 0.498 | 0.122 |
| Holt-Winters | 41.07 | 9.67 | 48.77 | 7.20 | 0.452 | 0.107 |
| SARIMA | 42.06 | 17.82 | 49.79 | 7.53 | 0.460 | 0.188 |
| **LightGBM** | **34.12** | **5.99** | **42.05** | **5.98** | **0.375** | **0.065** |

→ winner: **LightGBM**. The single holdout and the backtest disagree.

MASE by fold:

| Model | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Seasonal-naive | 0.476 | 0.539 | 0.377 | 0.396 | 0.394 | 0.449 | 0.660 | 0.690 |
| Holt-Winters | 0.571 | 0.310 | 0.362 | 0.369 | 0.553 | 0.551 | 0.521 | 0.377 |
| SARIMA | 0.457 | **0.850** | 0.590 | 0.450 | 0.328 | 0.282 | 0.441 | 0.286 |
| LightGBM | 0.488 | 0.314 | 0.362 | 0.318 | 0.309 | 0.377 | 0.447 | 0.387 |

Folds won: SARIMA 4, LightGBM 3, Holt-Winters 1. **SARIMA wins the most folds
and posts the worst single fold in the table (0.850).** LightGBM's spread is
about a third of SARIMA's and it has no bad fold.

## Backtest-design sensitivity (mean MASE)

| Model | single holdout | expanding, 3 folds | expanding, 8 folds | rolling-730, 8 folds |
|---|--:|--:|--:|--:|
| Seasonal-naive | 0.690 | 0.600 | 0.498 | 0.465 |
| Holt-Winters | 0.377 | 0.483 | 0.452 | 0.438 |
| SARIMA | **0.286** | **0.336** | 0.460 | 0.429 |
| LightGBM | 0.387 | 0.404 | **0.375** | **0.374** |
| **best** | SARIMA | SARIMA | **LightGBM** | **LightGBM** |

**Three folds — the brief's minimum — still picks the wrong model here.** The
expanding-vs-rolling choice does not change the ranking.

## Prediction intervals (nominal 80%)

Pooled over 8 folds × 14 days = 112 held-out points:

| | empirical coverage | calibration error | mean width | width as % of mean demand | point MAE |
|---|--:|--:|--:|--:|--:|
| **Conformal LightGBM** | **0.875** | +0.075 | **147.0** | **25.8%** | **34.12** |
| SARIMA native | 0.982 | +0.182 | 278.2 | 48.8% | 42.06 |

Coverage by fold:

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Conformal LightGBM | 0.857 | 1.000 | 0.929 | 1.000 | 0.857 | 0.857 | 0.643 | 0.857 |
| SARIMA native | 0.929 | 0.929 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Width by fold (units):

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Conformal LightGBM | 182.7 | 182.0 | 175.3 | 139.9 | 138.0 | 119.0 | 118.4 | 120.8 |
| SARIMA native | 283.8 | 289.7 | 269.9 | 265.1 | 260.6 | 272.5 | 283.9 | 299.7 |

**SARIMA wins on coverage and loses on the pair.** Reported coverage alone
would have selected the interval that spans half of typical daily volume.

Conformal half-widths on the development window: ±102.3 for steps 1–7, ±80.4
for steps 8–14 — **non-monotone**, i.e. within sampling noise at 6 calibration
blocks per bucket rather than a clean horizon-widening result.

## Final scorecard

| Model | MASE | MASE sd | WAPE % | MAE | Beats naive? | Folds won | 80% interval |
|---|--:|--:|--:|--:|---|--:|---|
| Seasonal-naive | 0.498 | 0.122 | 7.87 | 45.15 | yes | 0 | — |
| Holt-Winters | 0.452 | 0.107 | 7.20 | 41.07 | yes | 1 | — |
| SARIMA | 0.460 | 0.188 | 7.53 | 42.06 | yes | 4 | 98% cov @ 49% width |
| **LightGBM** | **0.375** | **0.065** | **5.98** | **34.12** | yes | 3 | **88% cov @ 26% width** |

**Recommendation: deploy LightGBM + split-conformal interval; keep SARIMA as a
monitored baseline.** The reasoning across history length, interpretability,
interval support and compute budget is in Section 7 of the notebook — the
accuracy margin is inside the fold-to-fold noise and is explicitly *not* the
basis for the decision.
