# Lending Club Loan Default — Profit-Based Funding Decisions

Predict whether a Lending Club loan will default, using **only information an
investor could see at listing time**, and fund a loan only when its
**expected profit is positive**.

- [`PLAN.md`](PLAN.md) — full project plan, phase by phase
- [`DECISIONS.md`](DECISIONS.md) — every decision, with the reasoning

> **Status:** Phase 4 complete. The model is trained, calibrated and scored on a held-out year.
> Full setup and run instructions land in Phase 8.

## Result on the 2015 test year (scored once)

| strategy | share funded | return per dollar | profit per $1B deployed |
|---|---|---|---|
| fund everything | 100% | +0.0655 | $65.5M |
| Lending Club grade A-B | 57.2% | +0.0718 | $71.8M |
| logistic regression (tuned threshold) | 71.3% | +0.0721 | $72.1M |
| **LightGBM (tuned threshold, p < 0.15)** | **66.9%** | **+0.0734** | **$73.4M** |
| LightGBM at grade A-B's selectivity | 56.1% | +0.0739 | $73.9M |

Returns are over the ~3-year loan term. The model earns about **12% more than
funding everything** and **2.2% more than Lending Club's own grade rule**, while
deploying more capital than that rule. Test ROC-AUC is 0.684.

**Cross-validated results** (expanding-window folds inside 2007-2013, tuned on log loss):

| model | log loss | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| Lending Club grade (baseline) | 0.36015 | 0.6232 | 0.1663 | 0.10473 |
| logistic regression | 0.35345 | 0.6618 | 0.2048 | 0.10331 |
| **LightGBM** | **0.35172** | **0.6672** | **0.2079** | **0.10300** |

**Headline numbers so far:** 621,022 matured 36-month loans, 13.95% default rate,
average realized profit +$1,029 per loan (+8.25% per dollar). Charged-off loans
still repay 63.5% of principal, so the loss given default is 0.365.

## Quick start

```bash
uv sync --extra dev          # or: pip install -r requirements.txt
uv run pytest                            # 18 tests
uv run python -m lending_club.data.ingest   # raw CSV -> parquet (~2 min)
uv run python -m lending_club.data.prepare  # clean + label + filter
jupyter lab notebooks/01_eda.ipynb          # Phase 1 EDA
dvc repro                                   # rebuild everything, end to end
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse the experiments
```

## Data

`accepted_2007_to_2018Q4.csv` (about 2.26M loans) from the Kaggle mirror
`wordsforthewise/lending-club`. It is tracked by DVC, not git — see
[D-028](DECISIONS.md#d-028-raw-data-via-manual-download-stored-as-uncompressed-csv)
for the file checksum and why.
