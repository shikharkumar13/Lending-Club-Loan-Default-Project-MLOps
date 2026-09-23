# Lending Club Loan Default — Profit-Based Funding Decisions

Predict whether a Lending Club loan will default, using **only information an
investor could see at listing time**, and fund a loan only when its
**expected profit is positive**.

- [`PLAN.md`](PLAN.md) — full project plan, phase by phase
- [`DECISIONS.md`](DECISIONS.md) — every decision, with the reasoning

> **Status:** Phase 7 complete. Three years of loans replayed as production traffic,
> with drift reports and a tested retraining trigger.
> Full setup and run instructions land in Phase 8.

## Result on the 2015 test year (scored once)

| strategy | share funded | return per dollar | profit per $1B deployed |
|---|---|---|---|
| fund everything | 100% | +0.0655 | $65.5M |
| Lending Club grade A-B | 57.2% | +0.0718 | $71.8M |
| logistic regression (tuned threshold) | 69.9% | +0.0722 | $72.2M |
| **LightGBM (tuned threshold, ~14.6% risk cut-off)** | **66.3%** | **+0.0735** | **$73.5M** |
| LightGBM at grade A-B's selectivity | 58.0% | +0.0738 | $73.8M |

Returns are over the ~3-year loan term. The model earns about **12% more than
funding everything** and **2.4% more than Lending Club's own grade rule**, while
deploying more capital than that rule. Test ROC-AUC is 0.684.

Every gap is significant on a 1,000-draw bootstrap of the test loans: LightGBM
beats the grade rule at equal selectivity by +0.0021 per dollar
(95% CI [+0.0015, +0.0027]). That measures sampling noise **within one test
year**; it says nothing about a different credit cycle, which is the larger
risk.

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
uv run pytest                            # 98 tests
uv run python -m lending_club.data.ingest   # raw CSV -> parquet (~2 min)
uv run python -m lending_club.data.prepare  # clean + label + filter
jupyter lab notebooks/01_eda.ipynb          # Phase 1 EDA
dvc repro                                   # rebuild everything, end to end
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse the experiments

# serve the model
uv run uvicorn lending_club.serving.app:app --reload    # http://localhost:8000/docs
docker build -t lending-club-api .
docker run -d --name lc-api -p 8001:8000 lending-club-api   # http://localhost:8001/docs
curl localhost:8001/health

# retrain end to end, with the promotion gate (~100s)
PREFECT_API_URL= PREFECT_SERVER_ALLOW_EPHEMERAL_MODE=true uv run python flows/retrain.py

# monitoring: replay 2016-2018 as monthly production batches (~4 min)
uv run python -m lending_club.monitoring.batches
uv run python -m lending_club.monitoring.drift
open reports/figures/drift.png reports/monitoring/html/2018-12.html
```

## Monitoring

The model is trained on 2007-2013 and reported on 2015. The 988,585 36-month
loans issued 2016-2018 are replayed as 36 monthly production batches, scored by
the deployed bundle.

**Most of them have no label, and that is the point.** A 36-month loan issued in
2018-06 has not finished by the 2018Q4 snapshot. The share of loans with a known
outcome falls from 99% to 3% across the window, and the default rate visible on
that shrinking subset swings from 14.3% to 20.5% and back to 12.6% purely from
label maturity — it would look exactly like a model going bad
([D-065](DECISIONS.md#d-065-the-labels-we-can-see-early-are-not-a-sample-we-can-score-on)).
So monitoring watches what is visible immediately: the inputs, and the model's
own output.

| What is measured | Result |
|---|---|
| Share of model inputs drifted vs. training | **43-60%, every month** |
| Prediction drift (model's own score distribution) | 0.03 → 0.28, breaching from 2017-12 |
| Share of loans the policy funds | stable, 0.63 → 0.73 |
| Alerts raised over 36 months | **3** (1 retrain, 2 investigate) |

Thirty-six months of breaching thresholds produce three alerts, not thirty-six:
a retrain fires once per episode, and a new category is reported once, not every
month ([D-067](DECISIONS.md#d-067-the-retraining-trigger-needs-persistence-not-a-spike)).
The two `investigate` alerts are real findings — `addr_state=ND` and
`home_ownership=ANY` are values that never appear in the training window.

![drift](reports/figures/drift.png)

## Data

`accepted_2007_to_2018Q4.csv` (about 2.26M loans) from the Kaggle mirror
`wordsforthewise/lending-club`. It is tracked by DVC, not git — see
[D-028](DECISIONS.md#d-028-raw-data-via-manual-download-stored-as-uncompressed-csv)
for the file checksum and why.
