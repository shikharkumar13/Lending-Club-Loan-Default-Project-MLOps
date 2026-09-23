# Lending Club Default Prediction: Project Plan

> Status: **Plan approved.** Build scope for round 1: see section 9.1.
> All decisions are summarized in [DECISIONS.md](DECISIONS.md), which is kept up to date as the project evolves.

---

## 1. The problem

An investor on a P2P lending platform sees a new loan listing and must decide
**whether to fund it**. Loans that are repaid earn interest. Loans that are
charged off lose most of the principal.

The goal is not simply "predict default with high AUC". It is:

> **For each new listing, estimate the probability of default. Fund the loan
> only if the expected profit is positive. Show that this policy earns more
> than simple baselines on loans the model has never seen, from a later time
> period.**

Three rules set this project apart from the typical Lending Club notebook:

1. **Only use data available at listing time.** No column may be known only
   after the loan was issued.
2. **Split by time, never randomly.** Train on older loans and test on newer
   ones, the way the model would actually be used.
3. **Judge the model by profit, not accuracy.** Pick the decision threshold
   that maximizes expected return.

---

## 2. Dataset

| Item | Detail |
|---|---|
| Source | Kaggle mirror: `wordsforthewise/lending-club` (Lending Club removed its official downloads in 2020) |
| File | `accepted_2007_to_2018Q4.csv.gz`, about 2.26M loans × about 151 columns |
| Also in download | `rejected_2007_to_2018Q4.csv.gz` (not used; out of scope) |
| Data dictionary | `LCDataDictionary.xlsx` (widely mirrored; we will commit a copy to `docs/`) |

> All of these numbers are from memory. They are checked for real in Phase 1.

### 2.1 Target definition

| `loan_status` value | Label |
|---|---|
| `Fully Paid` | 0 |
| `Charged Off`, `Default` | 1 |
| `Does not meet the credit policy. Status:Fully Paid` / `...Charged Off` | 0 / 1 (kept, flagged; decided in Phase 1) |
| `Current`, `In Grace Period`, `Late (16-30)`, `Late (31-120)` | **Dropped**: outcome not known yet |

### 2.2 Maturity bias (easy to miss)

The data is a snapshot taken at the end of 2018. A 36-month loan issued in
2017 can only show up as "finished" if it ended early. That usually means
it defaulted early or was prepaid. So recent loans look much riskier (or
safer) than they really are.

**Decision:** keep only loans that had enough time to reach an outcome:

- 36-month loans issued **on or before 2015-12**
- 60-month loans: **excluded from v1**. Only issue years up to 2013 would
  qualify, which leaves too little data. We can revisit this later.

### 2.3 Time-based split (36-month loans)

| Split | Issue dates | Purpose |
|---|---|---|
| Train | 2007-06 → 2013-12 | Fit models |
| Validation | 2014-01 → 2014-12 | Tune hyperparameters, calibrate, choose threshold |
| Test | 2015-01 → 2015-12 | Touched **once**, for the final numbers |
| "Production" simulation | 2016+ issued loans | Drift monitoring only (their labels are biased, see 2.2) |

### 2.4 Leakage control: an **allowlist**, not a blocklist

We list the columns that are allowed as features. Everything else is dropped
by default. This protects us from new or unfamiliar columns slipping in
unnoticed.

- **Allowed (examples):** `loan_amnt`, `term`, `int_rate`, `installment`,
  `grade`, `sub_grade`, `emp_length`, `home_ownership`, `annual_inc`,
  `verification_status`, `purpose`, `addr_state`, `dti`, `delinq_2yrs`,
  `fico_range_low/high`, `inq_last_6mths`, `open_acc`, `pub_rec`,
  `revol_bal`, `revol_util`, `total_acc`, `earliest_cr_line` (converted to
  credit-history length), `mort_acc`, `pub_rec_bankruptcies`
- **Forbidden (known only after issuance):** `total_pymnt*`, `total_rec_*`,
  `recoveries`, `collection_recovery_fee`, `last_pymnt_*`, `next_pymnt_d`,
  `out_prncp*`, `last_credit_pull_d`, `last_fico_range_*`, `hardship_*`,
  `settlement_*`, `debt_settlement_flag`, `pymnt_plan`
- **Excluded for other reasons:** `id`, `member_id`, `url`, `desc`,
  `emp_title` (free text; maybe in v2), `policy_code`, and any column that
  is mostly empty during the training years (many bureau fields were only
  added around 2015 and later)
> **Phase 1/2 update:** `mort_acc`, `tot_cur_bal`, `tot_coll_amt` and
> `application_type` were removed from this list after the data audit
> (D-030), and `features.groups` in `params.yaml` now decides which of the
> remaining columns become model inputs (D-035). `DECISIONS.md` is
> authoritative where it differs from this section.

- `int_rate` and `grade` **are allowed**. Lending Club assigns them before
  the listing goes live, so investors see them. Our model must therefore
  beat Lending Club's own grading, which makes it a tougher and more honest
  test.

Some forbidden columns (`total_pymnt`, `recoveries`) **are** used, but only
to compute the *realized profit* of each loan during evaluation. They never
become features.

---

## 3. Profit framing

For each loan *i* with principal `P`, installment `I` and 36-month term:

- **Realized return (for evaluation only):** `total_pymnt − P`. This uses
  what actually happened.
- **Expected return (the decision rule):**
  `E[return] = (1 − p_default) × gain_if_paid − p_default × loss_if_default`
  - `gain_if_paid ≈ 36·I − P`
  - `loss_if_default ≈ LGD × P`, where the loss given default (LGD) is
    estimated **on the training split only** from realized recoveries
- **Policy:** fund if `E[return] > 0`. Alternatively, pick the probability
  threshold that maximizes realized profit on the validation set.
- **Headline metric:** total and per-dollar return on the **test** set,
  compared with the baselines below.
- **Supporting metrics:** ROC-AUC, PR-AUC, Brier score, calibration curve.
  Calibration is important because the decision uses the probability value
  itself, not just its ranking.

### Baselines the model must beat

1. **Fund everything**
2. **Grade rule:** fund only grades A–B (a typical cautious investor)
3. **Logistic regression** on the same features
4. **LightGBM** (main model), calibrated with isotonic or Platt scaling on
   the validation set

---

## 4. Tech stack

It matches the Hotel Booking roadmap, so each tool gets used a second time
on a new problem.

| Concern | Tool | Why here |
|---|---|---|
| Env / deps | `uv` + `pyproject.toml` | Reproducible installs |
| Data versioning | DVC | The raw file is several GB, too big for Git; pipeline stages rerun only when their inputs change |
| Data processing | Polars (or pandas) + Parquet | 2M+ rows; Parquet is roughly 10× smaller and faster than CSV |
| Data validation | Pandera | Enforces the feature allowlist and valid value ranges |
| Modeling | scikit-learn, LightGBM | Baselines plus a strong model for tabular data |
| Explainability | SHAP | Needed to justify credit decisions |
| Experiment tracking | MLflow | Compare runs; model registry |
| Serving | FastAPI + Pydantic | `/predict` returns p_default, expected return and a decision |
| Containers | Docker | The same image runs locally and in CI |
| Orchestration | Prefect | A retraining flow: ingest → validate → train → evaluate → register |
| CI | GitHub Actions | Lint, tests, and a quick training run on a small sample |
| Monitoring | Evidently | Drift reports on the 2016+ "production" loans |

---

## 5. Repository layout (target)

```
lending-club-default/
├── PLAN.md
├── README.md
├── pyproject.toml
├── dvc.yaml                 # pipeline stages
├── params.yaml              # split dates, feature allowlist, model params
├── data/                    # DVC-tracked, git-ignored
│   ├── raw/
│   ├── interim/             # filtered + labeled parquet
│   └── processed/           # train / val / test feature tables
├── notebooks/               # EDA only, no pipeline logic
├── src/lending_club/
│   ├── data/                # ingest, filter, label, split
│   ├── features/            # feature engineering + schema
│   ├── models/              # train, calibrate, evaluate
│   ├── policy/              # profit calculation + threshold selection
│   └── serving/             # FastAPI app
├── flows/                   # Prefect flows
├── tests/
├── docs/                    # data dictionary, model card
├── Dockerfile
└── .github/workflows/
```

---

## 6. Phases

### Phase 0: Project setup
- `git init`, `uv init`, folder layout, `.gitignore`, pre-commit (ruff)
- `dvc init`; Kaggle CLI credentials check (`~/.kaggle/kaggle.json`)
- **Done when:** `uv run pytest` passes on an empty test and `dvc status`
  runs cleanly.

### Phase 1: Data acquisition & understanding
- Download via Kaggle CLI and track `data/raw/` with DVC
- Confirm row and column counts, the `loan_status` values and the issue-date
  range against the numbers in this plan
- EDA notebook: default rate by year / grade / term, missingness by year,
  distributions of key features
- Audit every column against the data dictionary and **finalize the
  allowlist** in `params.yaml`
- **Done when:** the allowlist is committed, each excluded column has a
  written reason, and the maturity cutoffs are confirmed from the data.

### Phase 2: Data pipeline (scripts, not notebooks)
- `ingest` → `filter_and_label` → `split` → `build_features`, as DVC stages
- Feature engineering: credit-history length, `term` → int,
  `emp_length` → int, `int_rate`/`revol_util` % strings → floats,
  income-to-loan ratio, FICO midpoint
- Pandera schema on the processed tables
- Unit tests: labeling logic, no post-issuance columns in the features,
  split boundaries respected
- **Done when:** `dvc repro` rebuilds everything from raw data and the
  leakage test passes.

### Phase 3: Modeling & experiment tracking
- Baselines (fund-all, grade rule, logistic regression) and LightGBM, all
  logged to MLflow
- Tune hyperparameters on validation, with rolling time-based folds inside
  the training period
- Calibration on validation; reliability plots logged as artifacts
- **Done when:** MLflow shows every run side by side and the best model is
  calibrated (Brier score and calibration curve logged).

### Phase 4: Profit evaluation & decision policy
- Estimate LGD from the training split
- Profit curve: realized return vs. threshold on validation; pick the best
  threshold
- **A single run on the test set:** return, % of loans funded and default
  rate among funded loans, for the model and every baseline
- SHAP global and per-loan explanations
- **Done when:** a results table and profit-curve chart exist and the chosen
  policy is frozen in `params.yaml`.

### Phase 5: Packaging & serving
- Save the full pipeline (preprocessing + model + calibrator + threshold) as
  one artifact and register it in MLflow
- FastAPI `/predict` (single loan and batch) and `/health`, with Pydantic
  models built from the allowlist
- Dockerfile; API tests using real rows from the test set
- **Done when:** `docker run` serves predictions matching the offline scores
  for the same inputs.

### Phase 6: Orchestration & CI/CD
- Prefect flow: ingest → validate → train → evaluate → register the model
  only if it beats the current production model on validation profit
- GitHub Actions: ruff, pytest, a training run on a 1% sample, Docker build
- **Done when:** a pull request runs every check automatically, and the
  Prefect flow runs end to end locally.

### Phase 7: Monitoring
- Treat 2016–2018 issued loans as a stream of monthly "production" batches
- Evidently reports: feature drift, drift in predicted default rate, and
  data-quality checks
- Define a retraining trigger (e.g. drift share > threshold for 2 months in
  a row)
- **Done when:** monthly drift reports are generated and the trigger logic is
  written down and tested. *(Done: 36 monthly batches, 988,585 loans, 18 tests;
  D-064..D-069.)*

### Phase 8: Documentation
- README: problem, approach, results table, architecture diagram, how to run
- Model card: intended use, data, metrics, limitations, fairness notes
  (e.g. `addr_state` as a possible proxy for protected attributes)
- **Done when:** someone new can clone the repo and follow the README to
  reproduce the test results. *(Done: README rewritten with a Mermaid
  architecture diagram and a verified setup path, MODEL_CARD.md with a measured
  fairness audit, `lending_club.fairness` + 5 tests; D-070, D-071.)*

---

## 7. Risks & open questions

| Risk | Mitigation |
|---|---|
| The CSV is several GB and heavy on RAM when loaded with pandas | Read with Polars, keep only needed columns, convert to Parquet once |
| The Kaggle mirror is not the official source | Record the file checksum with DVC and note where the data came from in the README |
| Lending Club ended retail P2P lending in 2020 | Present the project as historical and explain how it carries over to fintech and bank credit risk |
| Maturity bias leaves little recent labeled data | Documented limitation; 2016+ loans are used only for drift monitoring |
| Our LGD estimate is simplified | Sensitivity analysis: report profit across a range of LGD values |

**Choices with defaults (we can change them before Phase 1):**
- Include "does not meet credit policy" loans? *Default: include, with a flag.*
- Use 60-month loans? *Default: no for v1.*
- Polars or pandas? *Default: Polars for the pipeline, pandas where a
  library requires it.*

## 8. Out of scope for v1
- The rejected-loans file (reject inference)
- NLP on `desc` / `emp_title`
- Cloud deployment / Kubernetes / Terraform (practiced in the Hotel Booking
  roadmap)
- Portfolio optimization with budget constraints

---

## 9. Implementation details (added before the build)

This section fills in the specifics requested before coding starts: scope,
libraries, preprocessing, missing values, class imbalance, cross-validation
and metrics.

### 9.1 Scope of this build round

| In this round | Later rounds |
|---|---|
| Phase 0: setup | Phase 5: FastAPI + Docker serving |
| Phase 1: data loading + EDA | Phase 6: Prefect + GitHub Actions |
| Phase 2: leakage-safe preprocessing pipeline | Phase 7: Evidently monitoring |
| Phase 3: time-based CV, logistic regression vs. LightGBM, MLflow | |
| Phase 4: metrics, calibration, profit policy | |
| `tests/`, `requirements.txt`, README | |

This round produces a trained, evaluated and documented model. The
production layers get added on top later, so each round stays small enough
to understand.

### 9.2 EDA: notebook; pipeline: scripts

- **EDA → `notebooks/01_eda.ipynb`.** Exploring data is visual and
  back-and-forth, and a notebook shows plots next to the reasoning. Outputs
  are kept, so GitHub displays the charts to reviewers.
- **Everything the model depends on → `src/` scripts.** Notebooks run cells
  in any order, are hard to test and are hard to diff in Git. So the rule is
  that the notebook may *call* functions from `src/`, but no pipeline logic
  lives only in the notebook. Anything the notebook finds (e.g. "drop column
  X") becomes code or configuration in `params.yaml`.

### 9.3 Libraries

| Library | Role |
|---|---|
| `polars` + `pyarrow` | Fast, memory-efficient loading of the multi-GB CSV (lazy scan, reads only the columns we need) → Parquet |
| `pandas` | Handed to scikit-learn and SHAP, which expect pandas |
| `scikit-learn` | `Pipeline`, `ColumnTransformer`, imputers, encoders, logistic regression, calibration, metrics |
| `lightgbm` | Gradient-boosted trees; handles missing values and categoricals natively |
| `pandera` | Schema checks on the processed tables (types, ranges, allowlist) |
| `mlflow` | Records every experiment run (parameters, metrics, plots) |
| `shap` | Explains the model's predictions |
| `matplotlib`, `seaborn` | EDA and evaluation charts |
| `dvc` | Versions the data and defines the pipeline stages |
| `pyyaml` | Reads `params.yaml` |
| `pytest`, `ruff`, `jupyter` | Tests, linting, EDA |

**Dependency files:** `uv` + `pyproject.toml` define the dependencies (`uv`
also writes a lock file with exact versions). `requirements.txt` is
**generated** from that lock file with `uv export`, so plain
`pip install -r requirements.txt` works for anyone who doesn't use uv.
There's one source, and both install methods produce the same versions.

### 9.4 Preprocessing: two layers, split by whether they *learn* anything

This distinction is the core of leakage prevention.

**Layer A: stateless row-level cleaning (in `src/.../data`, run once on all
data).** Each row is transformed using only its own values, so doing this
before the split can't leak information:
- `"36 months"` → `36`, `"13.5%"` → `13.5`, `"10+ years"` → `10`,
  `"< 1 year"` → `0`
- `earliest_cr_line` + `issue_d` → `credit_history_months`
- `fico_mid = (fico_range_low + fico_range_high) / 2`
- Ratios: `loan_to_income = loan_amnt / annual_inc`,
  `installment_to_income = installment * 12 / annual_inc`
- Label creation and row filtering (section 2.1–2.2)

**Layer B: stateful transforms (inside an sklearn `Pipeline`, fitted on
training data only).** Anything that computes a statistic, like a median, a
category list, a mean or a percentile, is **learned from the training fold
only**. It is then applied unchanged to the validation and test data:
- Imputation values, outlier caps, one-hot category lists, scaling means and
  standard deviations
- Putting these in a `Pipeline` means cross-validation automatically refits
  them on each fold. It is impossible to accidentally fit them on the full
  data.

**Which columns to drop is also decided from training data only.** For
example, "drop if more than 50% missing" is measured on the 2007–2013 loans,
not on all years.

### 9.5 Missing values: the reason it's missing decides the fix

| Type | Example | Treatment |
|---|---|---|
| **Structural**: missing means "never happened" | `mths_since_last_delinq`, `mths_since_recent_inq` | Fill with a large sentinel value (e.g. 999 = "never") **plus** a `was_missing` indicator |
| **Sporadic**: a few rows missing at random | `revol_util`, `dti`, `emp_length` | Median imputation learned from training data, plus an indicator (being missing can itself be a signal) |
| **Didn't exist yet**: field added in later years | `open_acc_6m`, `il_util`, `all_util` (added around 2015 and later) | Drop them: they are empty for the whole training period |
| Categorical "n/a" | `emp_length = n/a` | Its own category, `unknown` |

LightGBM can handle missing values itself. We still use the same imputed
features for both models, so the comparison between them is fair. The
missing-value indicators keep the information about *which* values were
missing.

### 9.6 Categoricals and numeric features

- `grade` / `sub_grade`: **ordinal** encoding (A1 < … < G5). The order
  carries meaning, and one-hot encoding would throw it away.
- `home_ownership`, `verification_status`, `purpose`: one-hot. Rare levels
  (`NONE`, `ANY`, `OTHER`, and any level under 1% of rows in training) are
  merged into `other`, so unseen or rare values at prediction time don't
  break anything (`handle_unknown="infrequent_if_exist"`).
- `addr_state` (about 50 levels): one-hot for logistic regression; LightGBM
  gets the same encoding. Target encoding is left out of v1 because it
  easily causes leakage.
- Skewed amounts (`annual_inc`, `revol_bal`): `log1p` and capping at the
  training 99.5th percentile. Scaling is applied for logistic regression
  only; tree models don't need it.
- Highly correlated features (`int_rate` ~ `sub_grade`, `loan_amnt` ~
  `installment`) are kept. The correlation doesn't hurt LightGBM, and
  logistic regression uses L2 regularization, which keeps it stable.
  (Correlation does make SHAP explanations harder to read; we'll note that.)

### 9.7 Class imbalance: keep the real default rate; don't resample

The expected default rate is about 15–20% among completed 36-month loans
(checked in Phase 1). That is imbalanced, but not extremely.

**Decision: no SMOTE, no oversampling or undersampling.**
- The profit rule multiplies by `p_default`, so the probabilities have to be
  **calibrated**: when the model says 20%, about 20% of those loans should
  actually default. Resampling changes the default rate the model sees, which
  pushes all its probabilities up or down.
- SMOTE also invents synthetic borrowers by interpolating between real ones,
  which makes little sense for mixed categorical/numeric credit data.
- We handle the imbalance where it matters instead:
  1. **Metrics** that stay meaningful under imbalance (9.9), not accuracy
  2. **The decision threshold** is picked by profit, not fixed at 0.5
  3. `class_weight="balanced"` is tried as a **hyperparameter**, and
     calibration is redone afterwards if it's used

### 9.8 Cross-validation: expanding window by year

Random K-fold would train on 2013 loans and validate on 2010 loans. That is
*future information*: interest-rate environments and Lending Club's
underwriting rules changed over time, so random folds make scores look
better than they will be in practice. A custom year-based splitter is used,
because sklearn's `TimeSeriesSplit` splits by row count, not calendar year:

| Fold | Train on issue years | Validate on |
|---|---|---|
| 1 | 2007–2010 | 2011 |
| 2 | 2007–2011 | 2012 |
| 3 | 2007–2012 | 2013 |

- **Hyperparameter search:** randomized search (about 25 configurations for
  LightGBM; a grid over `C` and `class_weight` for logistic regression),
  scored by average log loss across the folds.
- **After CV:** refit the chosen settings on 2007–2013 → fit the calibrator
  on 2014 → pick the profit threshold on 2014 → score **2015 once**.
- The model is not refit on 2007–2014. The 2014 data is needed to fit the
  calibrator and choose the threshold, and reusing it for training would
  make both of those optimistic.

### 9.9 Metrics: which one we optimize, and why

| Where | Metric | Why |
|---|---|---|
| **Hyperparameter tuning (CV)** | **Log loss** | Our decision uses the probability *value*, not just the ranking. Log loss penalizes predictions that are confident and wrong, so it rewards models that both separate the classes well and are calibrated |
| Model comparison (reported) | ROC-AUC, **PR-AUC** | ROC-AUC measures overall ranking quality. PR-AUC focuses on the default class and doesn't look good just because most loans are repaid |
| Calibration check | Brier score, reliability curve | Shows whether "20% risk" actually means about 20% default |
| **Final business metric** | **Realized return on the test set** (total $, return per invested $, % funded, default rate among funded loans) | This is the investor's objective, compared against fund-all and the grade A–B rule |

**Why not accuracy:** predicting "no default" for every loan scores about
83% accuracy and funds everything, which is exactly the fund-all baseline.
**Why not F1:** F1 treats the two kinds of mistakes as roughly equally bad.
Here they aren't: a missed default loses about 60–70% of the principal,
while wrongly rejecting a good loan only loses about 15–25% in forgone
interest. The costs also differ loan by loan in dollar terms. The profit
calculation captures that asymmetry directly.

### 9.10 Leakage safeguards

1. A column allowlist (section 2.4), enforced by a Pandera schema
2. **An automated test** that fails if any forbidden column reaches the
   features
3. Time-based splits everywhere (CV, calibration, test)
4. All learned transforms inside the `Pipeline` and fitted per fold
5. Column-drop decisions and the loss-given-default estimate computed from
   training data only
6. The test set is scored once, by a separate `evaluate` step that runs
   after everything else is frozen

### 9.11 Updated project layout

```
Lending Club Loan Default Project/
├── PLAN.md
├── README.md                 # how to run + results + key decisions
├── pyproject.toml, uv.lock   # dependencies (source of truth)
├── requirements.txt          # exported from uv.lock
├── params.yaml               # dates, allowlist, model params, seeds
├── dvc.yaml                  # ingest → prepare → split → train → evaluate
├── data/{raw,interim,processed}/      # git-ignored, DVC-tracked
├── notebooks/01_eda.ipynb
├── src/lending_club/
│   ├── config.py             # loads params.yaml
│   ├── data/ingest.py        # CSV → parquet (allowlisted columns only)
│   ├── data/prepare.py       # Layer A cleaning, labels, maturity filter
│   ├── data/split.py         # time-based train/val/test + CV folds
│   ├── features/pipeline.py  # Layer B: sklearn ColumnTransformer
│   ├── features/schema.py    # Pandera schemas
│   ├── models/train.py       # CV + tuning + MLflow logging
│   ├── models/calibrate.py
│   ├── policy/profit.py      # LGD, expected return, threshold search
│   └── evaluate.py           # one-time test-set report
├── reports/{figures,metrics.json}
└── tests/                    # labeling, splits, leakage, pipeline, profit
```

### 9.12 Practical notes
- **Memory:** the raw CSV is several GB. Polars `scan_csv` with column
  selection keeps peak RAM low. Parquet is written once and everything after
  reads from it.
- **Reproducibility:** one `random_state` everywhere (from `params.yaml`),
  fixed library versions, and `dvc repro` rebuilds every output.
- **Kaggle credentials:** the download needs `~/.kaggle/kaggle.json`. If it
  isn't set up, the file can be downloaded from the browser into `data/raw/`
  by hand.
