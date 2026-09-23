# Model Card — Lending Club Loan Funding Model

A binary classifier that estimates the probability a 36-month Lending Club loan
will default, wrapped in a policy that turns that probability into a funding
decision.

| | |
|---|---|
| **Version** | `schema_version` 1, built 2026-09-23 |
| **Type** | LightGBM gradient-boosted trees + isotonic calibration |
| **Inputs** | 30 fields available at listing time → 73 engineered features |
| **Output** | risk score, calibrated default probability, expected return in dollars, `fund` / `decline` |
| **Artifact** | `artifacts/bundle/loan_decision_model.joblib` |
| **Served by** | FastAPI (`/predict`, `/predict/batch`), Docker image |
| **License / data** | Lending Club 2007–2018 public dataset; historical, not a live product |

---

## Intended use

**What this is for.** Ranking listed loans by expected profitability, so a
fixed amount of investor capital goes to the better half of the book. It is a
portfolio tool operating on loans Lending Club has *already approved and
priced*.

**What it is not for.**

- **Not a credit approval system.** It never sees a rejected applicant. Every
  loan in the training data passed Lending Club's own underwriting and was
  assigned a grade and an interest rate. The model cannot say whether someone
  should be granted credit; it says whether an already-priced loan is worth
  buying. Using it to deny credit would apply it to a population it has never
  observed — and, in the US, would put it under ECOA/Regulation B, which this
  model has not been assessed against.
- **Not for 60-month loans.** Excluded from v1; the maturity and prepayment
  behaviour differ.
- **Not a live-market model.** Lending Club ended retail P2P lending in 2020.
  Treat this as a historical study and a template.
- **Not for individual adverse-action decisions.** No reason codes are produced.

---

## How the decision is made

A probability is not a decision. The bridge is expected value:

```
E[return] = (1 − p) × expected_gain_if_repaid − p × loss_if_default
```

Both sides are measured on training loans only:

| Quantity | Value | Meaning |
|---|---|---|
| Loss given default (LGD) | **0.3657** | charged-off loans still repay ~63% of principal |
| Prepayment factor | **0.8288** | repaid loans return 83% of contractual interest; borrowers pay early |

**A finding worth stating plainly:** the pure expected-value rule funds 99.9% of
loans. At these interest rates almost every loan has positive expected value, so
"reject negative-EV loans" is not where the money is. The model's value is
**ranking under a capital constraint**. The deployed policy is therefore a
threshold tuned for return per dollar.

**The deployed rule has two conditions:**

```
fund  ⟺  risk_score < 0.14  AND  expected_return_usd > 0
```

The threshold (a calibrated default risk of ~14.6%) is what binds in practice.
The expected-value floor removed 10 of 187,752 funded loans on the test year —
all corrupt listings whose stated installment contradicts the amount and rate —
and exists so the service can never return a funding decision next to a
negative expected return (D-074).

---

## Training data

| Split | Loans | Window | Default rate |
|---|---|---|---|
| Train | 175,426 | 2007-06 → 2013-12 | 12.63% |
| Validation | 162,570 | 2014 | 13.73% |
| Test | 283,026 | 2015 | 14.89% |

Source: `accepted_2007_to_2018Q4.csv` (2,260,702 rows) from the Kaggle mirror
`wordsforthewise/lending-club`, SHA-256 `3eae03c2…d1f36a`.

**Filters applied**, and why they matter:

- **36-month loans only**, issued **2007-06 to 2015-12**. A loan issued in 2016
  has not finished by the 2018Q4 snapshot, so including it would mean either a
  missing label or a label biased towards loans that ended early. Verified:
  0.05% of 2015 loans are unresolved, versus 28% of 2016 and 60% of 2017.
- **Known outcomes only** (`Fully Paid`, `Charged Off`, `Default`).
- **Splits are by issue date, never random.** A random split would let the model
  learn from 2013 and be scored on 2010.

**Not represented in the data at all:** rejected applicants, borrowers outside
the US, anyone Lending Club did not market to, and — importantly — any
protected attribute. Race, sex, age and ethnicity are absent, which makes direct
bias measurement impossible (see *Fairness* below).

---

## Metrics

Tuning optimizes **log loss**, not accuracy or AUC. The funding rule multiplies
by the predicted probability, so a probability that is confidently wrong must be
punished; ranking alone is not enough. Accuracy is meaningless here — predicting
"never defaults" scores 85%.

**Cross-validated** (expanding-window folds inside 2007–2013):

| Model | Log loss | ROC-AUC | PR-AUC | Brier | Configs searched |
|---|---|---|---|---|---|
| Lending Club grade (baseline) | 0.36015 | 0.6232 | 0.1663 | 0.10473 | 1 |
| Logistic regression | 0.35345 | 0.6618 | 0.2048 | 0.10331 | 3 |
| **LightGBM** | **0.35172** | **0.6672** | **0.2079** | **0.10300** | 20 |

Paired across folds, LightGBM wins 3 of 3 (mean +0.00173 ± 0.00197) — but the
between-fold spread is 0.026, fifteen times the gap, and LightGBM received
nearly seven times the search budget. It is consistently ahead by a little, not
decisively better (D-079).

**2015 test year, scored once:** log loss 0.3992, ROC-AUC 0.6834, Brier 0.1206.

**Business result on the test year:**

| Strategy | Share funded | Return per dollar | Default rate of funded loans |
|---|---|---|---|
| Fund everything | 100% | +0.0655 | 14.9% |
| Lending Club grade A–B | 57.2% | +0.0718 | 9.1% |
| **This model (threshold 0.14)** | **66.3%** | **+0.0735** | **10.0%** |
| This model, matched to A–B selectivity | 58.0% | +0.0738 | 9.0% |

At equal selectivity the model beats the grade rule by **+0.0021 per dollar**
(1,000-draw bootstrap, 95% CI [+0.0015, +0.0027]). Against a plain logistic
regression its edge is only **+0.0006** [+0.0001, +0.0011] — the gap that
matters is model versus no model, not LightGBM versus logistic regression.
Those intervals measure sampling noise *within one test year*. They say nothing
about a different credit cycle, which is the larger risk.

**Most important features** (mean |SHAP|): `sub_grade`, `log_annual_inc`,
`int_rate`, `grade`, `dti`, `fico_mid`, `inq_last_6mths`,
`installment_to_income`. The model leans heavily on Lending Club's own grade and
rate — it is substantially *refining* their pricing, not replacing it.

---

## Fairness

The dataset contains no protected attributes, so disparate treatment cannot be
measured directly. What can be measured is whether outcomes differ across a
geographic feature that is a well-known proxy for race and income in the US:
`addr_state`.

**Finding 1 — funding rates differ sharply by state.** On the 2015 test year
(states with ≥1,000 loans), against an overall funding rate of 66.3%:

| Lowest | | Highest | |
|---|---|---|---|
| Nevada | 47.8% | Massachusetts | 74.0% |
| Florida | 57.1% | Texas | 73.4% |
| Mississippi | 59.0% | New Hampshire | 72.0% |

A 26-point spread. Some of this is genuine: those states really do default more.
But across states, the correlation between the model's mean score and the actual
default rate is only **0.51** — state-level score differences are only partly
justified by state-level risk differences.

The contrast with other grouping columns is what makes this stand out. A large
funding gap is not by itself a problem; an *unearned* one is:

| Grouped by | Funding rate spread | Score vs. actual outcome |
|---|---|---|
| `purpose` | 14.2% (small business) → 82.0% (credit card) | **+0.94** |
| `home_ownership` | 58.1% (rent) → 75.2% (mortgage) | +1.00 * |
| **`addr_state`** | 47.8% (NV) → 74.0% (MA) | **+0.51** |

`purpose` spreads far wider than `addr_state` and is almost entirely explained
by real differences in default. `addr_state` spreads less and is explained
half as well. (* `home_ownership` has only three groups, so a correlation of
1.00 across three points is close to meaningless — it is shown for contrast, not
as evidence.)

**Finding 2 — the model's accuracy is uneven across states.** Among loans the
policy *funded*, the realized default rate ranges from **5.8% (Oregon)** to
**14.3% (Arkansas)**, against 10.0% overall. Applicants in some states are
effectively held to a different bar than the headline threshold implies.

**Finding 3 — and the feature earns none of this.** `addr_state` carries only
**1.6%** of total SHAP weight. Removing it entirely costs nothing measurable:

| | Log loss (2014 validation) | ROC-AUC |
|---|---|---|
| With `addr_state` | 0.37692 | 0.6769 |
| Without `addr_state` | 0.37693 | **0.6770** |

A difference of 0.00001 in log loss, with AUC very slightly *better* without it.

**Recommendation: drop `addr_state`.** A feature that is a recognised proxy for
protected characteristics, that produces a 26-point spread in funding rates and
a 2.5× spread in realized error, and that contributes no measurable predictive
value, should not be in a credit model. This has **not** been applied to the
deployed bundle — removing it changes every published number and is a modelling
decision, not a documentation one. It is recorded here as the top open item
([D-071](DECISIONS.md#d-071-addr_state-is-a-proxy-that-earns-nothing)).

**Other proxy risks not assessed:** `home_ownership` and `emp_length` correlate
with wealth and age. `purpose` is self-reported and may correlate with
circumstance. None were tested for disparate impact.

---

## Limitations

1. **One credit cycle, one platform.** Training spans 2007–2013, including the
   financial crisis; the test year is 2015. Nothing here shows the model
   survives a *different* downturn.
2. **The test year is already stale.** Monitoring shows 43–60% of model inputs
   drifted in every month of 2016–2018. A model trained on 2007–2013 was
   already operating on a shifted population by January 2016.
3. **LGD is a single constant.** 0.3657 for every loan. In reality recovery
   varies by state, balance and time since default.
4. **Prepayment is a single constant.** 0.8288 applied uniformly; in practice
   better borrowers prepay more, which means the model's upside on the safest
   loans is probably overstated.
5. **Selection bias is unfixable here.** Only funded loans have outcomes. The
   model cannot learn about applicants Lending Club rejected.
6. **The source data contains internally inconsistent loans.** 1,015 (0.16%)
   have an `installment` that disagrees with the amortization of their amount,
   rate and term by more than 1%. The API now rejects such listings (D-073),
   but they remain in the training data, where they are a small amount of
   label-independent noise.
7. **Calibration is stepwise.** The isotonic calibrator produces 255 distinct
   values, with one block containing 39,792 loans. It is fine for reading a
   probability, but ranking policies use the raw score, because ties inside a
   block would decide who gets funded arbitrarily.
8. **No fairness constraint is implemented.** The policy maximizes return per
   dollar, full stop.
9. **Returns are nominal**, not discounted, and ignore platform fees, taxes and
   the cost of capital.

---

## Monitoring and maintenance

- **2016–2018 loans are replayed as 36 monthly production batches** (988,585
  loans), scored by the deployed bundle.
- **Labels are not used to monitor quality.** Outcome coverage falls from 99% to
  3% across the window, and the default rate visible on that shrinking subset
  swings 14.3% → 20.5% → 12.6% purely from label maturity. Charting it would
  look exactly like a model failing.
- **Watched instead:** input drift (pinned Wasserstein / Jensen-Shannon),
  prediction drift, funding rate, and data-quality checks.
- **Retraining trigger:** drifted share > 0.30 or prediction drift > 0.15, for
  2 consecutive months. Data-quality breaks raise a separate `investigate`
  alert, because retraining does not fix a broken feed.
- **Promotion gate:** a retrained model replaces production only if it improves
  return per dollar by more than 0.0005 on validation data. The test year is
  never consulted by the gate.

**Known data findings:** `addr_state=ND` and `home_ownership=ANY` appear in
production but never in training. Neither breaks the service — unseen levels map
to "infrequent" — but both are reported.

---

## Reproducing this card

```bash
uv sync --extra train --extra dev
dvc repro                       # rebuild every number above from the raw CSV
```

Every figure here comes from `reports/metrics.json`, `reports/cv_results.json`,
`reports/monitoring/drift.json` or `reports/fairness.json`. The fairness tables
are regenerated by:

```bash
uv run python -m lending_club.fairness
```

with the audit logic covered by `tests/test_fairness.py`. The `addr_state`
ablation in *Finding 3* is a one-off experiment, not a pipeline stage; the
recipe is in [D-071](DECISIONS.md#d-071-addr_state-is-a-proxy-that-earns-nothing).
