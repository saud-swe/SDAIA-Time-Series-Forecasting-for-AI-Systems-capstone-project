# Backtested Forecasting Report — Riyadh / Grocery Daily Demand

**Capstone project — SDAIA Academy, *Time Series Forecasting for AI Systems***

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saud-swe/SDAIA-Time-Series-Forecasting-for-AI-Systems-capstone-project/blob/main/capstone_forecasting_report.ipynb)

| | |
|---|---|
| **Author** | Saud |
| **Training programme** | SDAIA Academy — Time Series Forecasting for AI Systems |
| **Cohort** | 15 September 2026 |
| **Deliverable** | [`capstone_forecasting_report.ipynb`](capstone_forecasting_report.ipynb) — one notebook, run top to bottom, output committed |
| **Course repository** | <https://github.com/MohammadYusif/time-series-forecasting-ai-systems> |
| **SDAIA Academy** | <https://github.com/SDAIAAcademy> |

---

## The project

This project takes one daily retail demand series from raw CSV to a
**validated, uncertainty-aware forecast**, and then answers the question that
actually decides a deployment: *which model family would I put into
production for this series, and how much should anyone trust that choice?*

Four model families are compared — a seasonal-naive baseline, Holt-Winters
exponential smoothing, SARIMA, and a LightGBM with engineered temporal
features — under an **8-fold expanding walk-forward backtest** with a 14-day
horizon, scored on scale-free metrics, and extended to **80% prediction
intervals** scored on coverage *and* width together.

The work is deliberately structured so that the interesting results are the
ones that contradict the easy answer. Three of them:

- **A single 14-day holdout picks a different winner than the 8-fold
  backtest.** So does a 3-fold backtest. The single split is not a weaker
  version of the walk-forward result — on this series it is simply wrong.
- **SARIMA's native 80% prediction interval achieves 98% empirical
  coverage** — and is nearly twice as wide as the conformal alternative.
  Judged on coverage alone it wins; judged honestly it is close to useless.
- **The information criteria and the residual diagnostics disagree about the
  model order**, and the notebook resolves that conflict explicitly rather
  than reporting whichever number came first.

## The dataset, and why this one

`data/retail_demand.csv` from the course repository, filtered to
**`region = Riyadh`, `category = Grocery`** — 1096 complete daily
observations, 2023-01-01 to 2025-12-31.

The capstone brief offers four datasets. I picked this one because it is the
only one that exercises **every** rubric section at full strength rather than
forcing a section to be argued around:

| | |
|---|---|
| **Two seasonal periods** (weekly ≈ 40% swing, plus yearly) | gives decomposition, ACF/PACF and SARIMA real work to do |
| **~3 years of daily history** | enough for a GBM *and* enough for a classical model, so the comparison in Section 7 is a genuine contest rather than a walkover |
| **Sporadic promo and moving-holiday shocks** | makes the residuals non-Gaussian, which is what makes the probabilistic section a real test instead of a formality |
| **A smooth trend with no structural break** | makes expanding-vs-rolling a decision that has to be *argued*, not defaulted |
| **No zero values** | forces an honest metric argument: MAPE is well-defined here, and I still do not rank on it — and I say why |

The trade-off I accepted: `workforce_demand.csv`'s structural break would have
made the backtest-design discussion richer, and `intermittent_demand.csv`
would have made the metric argument a slam-dunk. Section 7's "what would
change this recommendation" table addresses both directly.

**All four course datasets are synthetic**, generated deterministically by
`data/generate_series.py` (`SEED = 20260912`). Nothing in this report
describes a real retailer, and no figure here should be quoted as an
observation about Saudi grocery demand.

## Headline results

8-fold expanding walk-forward, 14-day horizon, 112 held-out days:

| Model | MASE ↓ | MASE sd across folds | WAPE % | Folds won (of 8) | 80% interval |
|---|--:|--:|--:|--:|---|
| Seasonal-naive (baseline) | 0.498 | 0.122 | 7.87 | 0 | — |
| Holt-Winters (add trend, mul seasonal) | 0.452 | 0.107 | 7.20 | 1 | — |
| SARIMA(1,0,1)(1,1,1)[7] on log | 0.460 | 0.188 | 7.53 | 4 | 98% cov @ 49% width |
| **LightGBM + conformal** | **0.375** | **0.065** | **5.98** | 3 | **88% cov @ 26% width** |

**Recommendation: deploy the LightGBM with a split-conformal interval; keep
SARIMA as a monitored baseline.** The point-accuracy margin is inside the
fold-to-fold noise and is *not* the reason. The reasons are that LightGBM is
by far the most consistent (a third of SARIMA's variance, and no bad fold),
that its conformal interval is near-nominal at half the width, and that it
extends to all six region × category pairs as one global model. Full
reasoning across history length, interpretability, interval support and
compute budget is in Section 7 of the notebook.

## How to run it

### Colab — the expected path, nothing to install

Click the badge at the top. Run the first cell and then Runtime → Run all.
The setup cell installs the packages and downloads `common/metrics.py`,
`common/backtest.py` and the dataset from the course repository. **No API key,
no GPU, no paid service, no account beyond Colab itself.** Full run is about
90 seconds.

### Locally

```bash
git clone https://github.com/saud-swe/SDAIA-Time-Series-Forecasting-for-AI-Systems-capstone-project.git
cd SDAIA-Time-Series-Forecasting-for-AI-Systems-capstone-project
pip install -r requirements.txt
jupyter lab capstone_forecasting_report.ipynb
```

The notebook is committed **with all output already captured**, so it reads
end to end without being run.

## Repository layout

```
├── capstone_forecasting_report.ipynb   ← the deliverable: one notebook, 7 sections
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── build/
│   ├── build_capstone.py               ← generates the notebook (cell structure as code)
│   └── execute.py                      ← executes it and verifies every cell has real output
└── docs/
    ├── TECHNICAL.md                    ← methods, design decisions, and why each was made
    ├── RESULTS.md                      ← every number this project reports, in one place
    ├── RUBRIC_MAP.md                   ← rubric section → where the evidence lives
    └── SUBMISSION_CHECKLIST.md         ← the brief's pre-submission checklist, verified
```

**On `build/`:** the notebook is generated from `build/build_capstone.py`
rather than hand-edited, the same way the course repository generates its own
labs from `_build/build_labN.py`. That keeps cell structure reproducible and
makes `build/execute.py`'s verification meaningful — it asserts that every
code cell has a non-null `execution_count`, that no cell produced an error,
and that no markdown cell violates Quarto's `---` rule. The *deliverable* is
still exactly one notebook, as the brief requires.

## Notes on reproducibility

- `RNG_SEED = 20260912` throughout — the same seed the course's data
  generator uses.
- No data is vendored or hand-edited; everything is fetched at run time from
  the course repository by the standard `fetch()` helper.
- The committed output is the output of a single top-to-bottom run.
- No credential, token or API key appears anywhere in the notebook or in this
  repository's git history.

## Licence

[MIT](LICENSE) — coursework, freely reusable.
