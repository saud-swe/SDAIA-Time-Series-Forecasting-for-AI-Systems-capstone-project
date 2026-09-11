# Rubric map

Where the evidence for each rubric section lives, and — for each row — what
the brief says "full marks" looks like, what this project did, and how it
avoided the mistake the brief flags as most likely to cost points.

Rubric per `capstone.qmd`: **100 points, draft pass mark 60/100**. Section
weights are the brief's own draft weights.

---

## 1. Time Series Structure & Diagnostics — 15 pts

**Where:** notebook §0 "The series" and §1 "Structure and diagnostics".

| | |
|---|---|
| Decomposition | STL on both raw and log scale, 4-panel plot, with Hyndman–Athanasopoulos seasonality/trend strength measures computed (0.626 / 0.745 on log) |
| ACF/PACF | 2×2 plot — ACF and PACF before and after seasonal differencing — plus printed ACF values at the seasonal lags |
| Stationarity | ADF **and** KPSS across five transforms, in one table |
| Differencing | seasonal difference at lag 7 applied; `d=1` tested in §2's grid and rejected |

**Interpreted in prose, not just plotted.** Each of the three has its own
"reading" markdown cell drawing a conclusion: the series is multiplicative
(so everything classical is fitted on `log`); the lag-7 PACF spike rules out a
plain ARIMA; seasonal differencing is required.

**The flagged mistake — "running the ADF test, printing a p-value, and never
saying what it means"** — is the subject of a dedicated subsection. ADF
*rejects* the unit-root null on the raw series (p = 0.0023), and KPSS agrees
it is stationary. The notebook explains why taking that at face value would be
wrong: both tests examine the *level*, neither has machinery for a 7-day
cycle, and a series that swings 40% weekly on a fixed schedule can be
level-stationary while remaining almost entirely unexplained. Seasonal
differencing is then justified by the ACF collapsing, not by a p-value.

## 2. Classical Forecasting Models — 15 pts

**Where:** notebook §2.

| | |
|---|---|
| Model | SARIMA(1,0,1)(1,1,1)[7] with constant, on `log(units_sold)` |
| Second family | Holt-Winters (additive trend, multiplicative seasonal, m=7) |
| Order rationale | 7-candidate grid derived from the §1 ACF/PACF read, scored by AIC **and** BIC, then a second-stage comparison of the two finalists on residual adequacy |
| Residual diagnostic | Ljung-Box at lags 7/14/21/28 with explicit verdicts, plus residual time plot, histogram, ACF and Q-Q plot |

**Beyond the minimum:** AIC/BIC and the residual diagnostic **disagree**. AIC
and BIC both favour the simpler `(1,0,1)(0,1,1)[7]`; that model leaves
significant Ljung-Box autocorrelation at **lag 7** (p = 0.046), the dominant
structure in the series. The notebook fits both finalists side by side, shows
the conflict in a table, and resolves it explicitly in favour of residual
adequacy — with the reasoning stated.

**The flagged mistake — "default or arbitrarily-chosen orders and never
checking whether the residuals still have structure"** — is inverted here: the
residual check is what *drives* the order choice rather than following it. And
when Ljung-Box still rejects at lags 14–28 for the chosen model, that is
**reported as a genuine inadequacy** and diagnosed (a missing holiday
regressor, not a misspecified ARMA order) rather than patched by stacking
terms until the test passes.

## 3. ML/GBM Forecasting & Feature Engineering — 15 pts

**Where:** notebook §3.

| | |
|---|---|
| Model | LightGBM, fixed hyper-parameters, `random_state = RNG_SEED` |
| Lags | `lag_1`, `lag_7`, `lag_14`, `lag_28` |
| Rolling | `roll_mean_7`, `roll_std_7`, `roll_mean_28` |
| Calendar | `dow`, `is_weekend` (Fri/Sat, KSA), `month`, `doy_sin`, `doy_cos` |

**Leakage awareness is demonstrated, not asserted.** Three layers:

1. **Structural** — every lag/rolling column derives from `series.shift(1)` or
   earlier; `make_features` is called inside each fold on training data only;
   no global statistic or scaler exists anywhere.
2. **Explained** — a markdown cell naming the three traps specifically (the
   rolling window that includes today; scaling before splitting; multi-step
   forecasting with real lags) and what closes each.
3. **Verified** — a mechanical audit re-derives four features for sampled
   dates from strictly-past data and asserts they match and that no feature
   equals its own target; the recursive loop asserts alignment at every step.

**The flagged mistake — "a feature built using information that wouldn't be
available at forecast time, especially in multi-step recursive forecasts"** —
is addressed head-on: forecasting is recursive, and the notebook **logs the
lag lineage per step** (0 of 4 lags synthetic at step 1, 2 of 4 by step 8) to
make concrete why error grows with horizon and why a 1-step-ahead score would
overstate the model.

## 4. Backtesting Framework & Time-Based Validation — 20 pts

**Where:** notebook §4.

| | |
|---|---|
| Harness | `common/backtest.py` — `expanding_window_splits` + `run_backtest`, used as fetched |
| Folds | **8** (brief requires ≥ 3), 14-day horizon, 730-day minimum training window |
| Window type | expanding, **justified from this series' own structure** (smooth monotone STL trend, no break) and contrasted with `workforce_demand.csv`, where a rolling window would be correct |
| Verification | overlap, contiguity, monotone expansion, horizon and final-fold-endpoint all asserted in code |

**Beyond the minimum:** the expanding-vs-rolling argument is *tested* — the
whole comparison is re-run under a fixed 730-day rolling window and the null
result reported. The notebook also reports the same comparison under four
backtest designs (single holdout / 3 folds / 8 folds / rolling 8 folds).

**The flagged mistake — "a single train/test split presented as if it were a
backtest"** — is the organising argument of the section. The single 14-day
holdout is run *first*, deliberately, and shown to pick **SARIMA**; the 8-fold
walk-forward picks **LightGBM**. A 3-fold backtest — the brief's own minimum —
agrees with the single split and is also wrong. The fold-boundary leak the
brief mentions is excluded by assertion.

## 5. Evaluation Metrics & Reporting — 10 pts

**Where:** notebook §5.

MAE, RMSE, MAPE, WAPE and MASE(sp=7), all from `common/metrics.py`, computed
**per fold** as well as pooled, with fold-to-fold standard deviations
reported alongside every mean.

**Scale-free metric choice justified by the series' own shape.** MASE is the
headline (scaled against seasonal-naive, so the number answers "is this worth
deploying"); WAPE is the secondary.

**The flagged mistake — "reporting MAPE on a series with zeros"** — required
an honest inversion here: **this series has no zeros** (minimum 397), so MAPE
is well-defined, and it is reported rather than suppressed. The notebook still
declines to *rank* on it, and says why in terms that survive the absence of
zeros: MAPE's per-row denominator penalises over-forecasting a quiet day more
than under-forecasting a promo day by the same units, which quietly rewards a
systematically low forecast on a spike-driven series. The
zeros-break-MAPE case is addressed explicitly by reference to
`intermittent_demand.csv`.

## 6. Probabilistic Forecasting & Prediction Intervals — 15 pts

**Where:** notebook §6.

| | |
|---|---|
| Method (a) | split-conformal on LightGBM — 6 × 14-day calibration blocks per fold, finite-sample quantile `ceil((n+1)(1−α))/n`, half-widths per horizon bucket |
| Method (b) | SARIMA's native parametric interval |
| Nominal level | 80% |
| Scoring | `coverage()` **and** `interval_width()` together, per fold and pooled, plus a coverage-vs-width scatter |

**The flagged mistake — "reporting coverage alone; an absurdly wide interval
covers everything and that number alone hides it"** — is the section's central
finding rather than something merely avoided. SARIMA's native interval reaches
**98.2%** empirical coverage against a nominal 80% — the best coverage in the
comparison — while being **nearly twice as wide** (48.8% of mean daily demand
vs the conformal interval's 25.8%). Coverage alone would have selected it. The
cause is traced back to §2's residual diagnostics: the parametric interval
assumes Gaussian uncorrelated residuals, which Ljung-Box and the Q-Q plot
already showed to be false.

Two limitations are reported rather than smoothed over: the conformal interval
is conservative (87.5% vs 80% nominal), and its step-bucket half-widths came
out non-monotone in horizon.

## 7. Model Comparison & Documentation — 10 pts

**Where:** notebook §7, plus this repository's `docs/`.

The written recommendation reasons from all four axes named in
`day3/06_model_comparison.qmd` — **history length, interpretability, interval
support, compute budget** — each with a stated verdict, including one axis
(history length) that explicitly *does not* separate the models here and one
(interval support) that resolves against the naive reading.

**The flagged mistake — "naming a winner by accuracy alone"** — is refused
explicitly. The notebook states that LightGBM's mean-MASE lead (~0.08) is
*smaller than the fold-to-fold standard deviation* of the models it beats, and
that SARIMA actually **wins more individual folds (4 of 8)** than LightGBM
does. The recommendation therefore rests on consistency (a third of SARIMA's
variance, no bad fold), interval quality, and extensibility to the other five
series — not on the accuracy ranking. A "what would change this
recommendation" table lists five concrete conditions that would flip it.

---

## GitHub requirements (evaluated on every SDAIA Academy project)

| Requirement | Where |
|---|---|
| Active GitHub account | trainee's own |
| Notebook uploaded to a repository, documented and kept updated | this repository |
| Clear, comprehensive project description | `README.md`, plus the repository "About" blurb in `docs/SUBMISSION_CHECKLIST.md` |
| Professional README — project idea, dataset choice and why, how to run | `README.md`, with the Colab badge as the expected path |
| Proper technical documentation beyond the notebook's markdown | `docs/TECHNICAL.md`, `docs/RESULTS.md`, this file |
| Good Git practices — meaningful incremental commits, sane structure, `.gitignore` | ten commits, one per stage of the work; layout in `README.md`; `.gitignore` excludes secrets, caches and generated files |
| Training programme name and cohort dates | `README.md` header and the notebook's header cell |
| Link to SDAIA Academy's GitHub | `README.md` header and the notebook's header cell |
