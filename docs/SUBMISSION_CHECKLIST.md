# Pre-submission checklist

The checklist from `capstone.qmd`, with the state of this repository against
each item.

## Status

Everything on the brief's checklist is done. The two items that previously
needed the author's input — GitHub username and cohort dates — are filled in:

| | |
|---|---|
| **Authors** | Saud ALajmi, Fahad Alanazi, Hamed Aldkhyyal |
| **GitHub** | [`saud-swe`](https://github.com/saud-swe) |
| **Programme** | SDAIA Academy — Time Series Forecasting for AI Systems |
| **Cohort** | 15 September 2026 |
| **Repository** | `SDAIA-Time-Series-Forecasting-for-AI-Systems-capstone-project` |

To confirm the notebook still parses after any edit:

```bash
python -c "import json; json.load(open('capstone_forecasting_report.ipynb')); print('notebook JSON valid')"
```

## Suggested GitHub "About" description

Paste this into the repository's About field (the brief asks for a clear
project description in the repository, and the About blurb is the first thing
a reviewer sees):

> Backtested demand-forecasting report for a daily retail series (SDAIA
> Academy capstone). Compares seasonal-naive, Holt-Winters, SARIMA and
> LightGBM under an 8-fold walk-forward backtest with conformal prediction
> intervals — and shows where a single train/test split gets the answer wrong.

Suggested topics: `time-series` `forecasting` `sarima` `lightgbm`
`conformal-prediction` `backtesting` `sdaia`

---

## The brief's checklist

- [x] **Your name is in the README and the notebook header.** Both — README
      header table and the notebook's second markdown cell.
- [x] **The notebook runs top to bottom in a fresh Colab runtime with no
      manual setup beyond the standard environment-setup cell.** The setup
      cell is copied from the course labs' own shape: `_pip_install` for its
      package list, then the `fetch()` helper that falls back to
      `raw.githubusercontent.com` when no local checkout exists. Verified by a
      clean top-to-bottom execution in a container with only the packages that
      cell installs.
- [x] **Every cell has real, captured output saved in the file.**
      `build/execute.py` asserts that every code cell has a non-null
      `execution_count` and that no cell produced an error output. Last run:
      33 code cells, 0 null, 0 errors.
- [x] **The backtest has at least 3 folds and no fold's training data overlaps
      its own test window.** 8 folds. Overlap, contiguity, monotone expansion,
      horizon and final-fold endpoint are all asserted in the notebook itself,
      so a regression would fail the run rather than pass silently.
- [x] **At least one scale-free metric (MASE or WAPE) is reported, and MAPE is
      not used alone on a series with zeros.** MASE(sp=7) is the headline
      metric and WAPE the secondary. This series has no zeros, so MAPE is
      well-defined and is reported — but not ranked on, with the reasoning
      given in §5.
- [x] **A coverage number and a width number are both reported for the
      interval forecast.** Both, for two interval methods, per fold and
      pooled, plus a coverage-vs-width scatter. The gap between them is §6's
      main finding.
- [x] **The model-comparison section is written in markdown cells in this same
      notebook, and reasons from more than one axis.** §7, four axes (history
      length, interpretability, interval support, compute budget), plus a
      table of five conditions that would change the recommendation.
- [x] **The repository has a README, a `.gitignore`, meaningful commit
      history, and states the training programme and cohort dates.** README
      and `.gitignore` present; incremental commits following the
      actual order of the work rather than one dump commit at the deadline; programme and cohort dates
      both stated in the README header and the notebook header.
- [x] **No API key or credential appears anywhere in the notebook or the git
      history.** Nothing in this project authenticates to anything. The
      `.gitignore` excludes `.env`, `*.key`, `*.pem`, credentials and token
      files as a standing guard. To confirm before pushing:

      ```bash
      git log -p | grep -inE "api[_-]?key|secret|password|bearer |sk-[a-zA-Z0-9]{20}|ghp_"
      ```

      Verified: the only matches are this checklist's own wording, the
      `.gitignore` comment header, and the commit message that introduced it —
      the scan matches the *word*, not a credential. Read the hits rather than
      trusting the count.

## Things the brief encourages but does not score

Worth doing once the repository is up: star and follow strong Saudi
repositories and accounts, contribute to open source, engage through forks,
pull requests and issues, and share the project with the community.
[SDAIA Academy's GitHub](https://github.com/SDAIAAcademy) is a reasonable
starting point.
