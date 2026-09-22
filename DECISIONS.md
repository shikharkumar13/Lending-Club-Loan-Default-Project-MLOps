# Decision Log

A running record of every significant decision in this project: **what** we
decided, **why**, and **what we rejected**. New decisions are added at the
bottom as the project moves forward. When a decision changes, the old entry is
marked *Superseded* and linked to its replacement. Old entries are never
deleted, so the history stays readable.

> **Status values:** `Accepted` · `Proposed` (not confirmed yet) ·
> `Superseded by D-xxx` · `Rejected`
>
> For full technical detail, see [PLAN.md](PLAN.md). This file is the short,
> readable summary.

---

## Index

| ID | Decision | Area | Status | Date |
|---|---|---|---|---|
| [D-001](#d-001-project-lending-club-default-with-profit-based-decisions) | Lending Club default prediction with profit-based decisions | Scope | Accepted | 2026-09-22 |
| [D-002](#d-002-dataset-source-kaggle-mirror) | Dataset: Kaggle mirror `wordsforthewise/lending-club` | Data | Accepted | 2026-09-22 |
| [D-003](#d-003-target-definition) | Target = Charged Off/Default vs. Fully Paid; unfinished loans dropped | Data | Accepted | 2026-09-22 |
| [D-004](#d-004-handle-maturity-bias-36-month-loans-issued--2015) | Keep only 36-month loans issued ≤ 2015 | Data | Accepted | 2026-09-22 |
| [D-005](#d-005-time-based-trainvalidationtest-split) | Time-based split: train 2007–13, validation 2014, test 2015 | Validation | Accepted | 2026-09-22 |
| [D-006](#d-006-feature-allowlist-not-blocklist) | Feature allowlist (listing-time data only) | Leakage | Accepted | 2026-09-22 |
| [D-007](#d-007-int_rate-and-grade-are-allowed-features) | `int_rate` and `grade` are allowed features | Features | Accepted | 2026-09-22 |
| [D-008](#d-008-post-issuance-columns-only-for-evaluation) | Post-issuance columns used only to compute realized profit | Leakage | Accepted | 2026-09-22 |
| [D-009](#d-009-decide-by-expected-profit) | Fund a loan if expected profit > 0; LGD estimated on train | Business | Accepted | 2026-09-22 |
| [D-010](#d-010-baselines-to-beat) | Baselines: fund-all, grade A–B rule, logistic regression | Evaluation | Accepted | 2026-09-22 |
| [D-011](#d-011-build-round-1-scope-phases-04) | Round 1 builds Phases 0–4 only | Scope | Accepted | 2026-09-22 |
| [D-012](#d-012-same-mlops-stack-as-hotel-booking-roadmap) | Same MLOps tool stack as the Hotel Booking roadmap | Tooling | Accepted | 2026-09-22 |
| [D-013](#d-013-eda-in-a-notebook-pipeline-in-scripts) | EDA in a notebook; pipeline in `src/` scripts | Structure | Accepted | 2026-09-22 |
| [D-014](#d-014-uv--pyprojecttoml-with-exported-requirementstxt) | `uv` + `pyproject.toml`; `requirements.txt` exported | Tooling | Accepted | 2026-09-22 |
| [D-015](#d-015-polars-for-data-pandas-at-the-sklearn-boundary) | Polars for data processing, pandas for scikit-learn | Tooling | Accepted | 2026-09-22 |
| [D-016](#d-016-two-layer-preprocessing) | Two-layer preprocessing (row-by-row vs. learned-from-data) | Leakage | Accepted | 2026-09-22 |
| [D-017](#d-017-missing-values-handled-by-why-theyre-missing) | Missing values: treat by why they're missing, plus indicators | Features | Accepted | 2026-09-22 |
| [D-018](#d-018-categorical-encoding) | Ordinal grades, one-hot others, rare levels grouped | Features | Accepted | 2026-09-22 |
| [D-019](#d-019-numeric-transforms) | log1p + cap skewed amounts; scale for logistic regression only | Features | Accepted | 2026-09-22 |
| [D-020](#d-020-keep-correlated-features) | Keep correlated features | Features | Accepted | 2026-09-22 |
| [D-021](#d-021-no-resampling-for-class-imbalance) | No SMOTE or resampling for class imbalance | Modeling | Accepted | 2026-09-22 |
| [D-022](#d-022-expanding-window-cross-validation-by-year) | Cross-validation by year, always training on earlier years | Validation | Accepted | 2026-09-22 |
| [D-023](#d-023-algorithms-logistic-regression-vs-lightgbm) | Logistic regression vs. LightGBM | Modeling | Accepted | 2026-09-22 |
| [D-024](#d-024-tuning-calibration-and-threshold-procedure) | Tune in CV → calibrate on 2014 → pick threshold on 2014 → test 2015 once | Validation | Accepted | 2026-09-22 |
| [D-025](#d-025-metrics) | Tune by log loss; final score is realized profit | Evaluation | Accepted | 2026-09-22 |
| [D-026](#d-026-does-not-meet-credit-policy-loans-kept-with-a-flag) | "Does not meet credit policy" loans kept, with a flag | Data | Accepted | 2026-09-22 |
| [D-027](#d-027-out-of-scope-for-v1) | Out of scope for v1: rejected loans, text NLP, cloud | Scope | Accepted | 2026-09-22 |
| [D-028](#d-028-raw-data-via-manual-download-stored-as-uncompressed-csv) | Raw data downloaded by hand; stored as uncompressed CSV in `data/raw/` | Data | Accepted | 2026-09-22 |

---

## Decisions

### D-001: Project: Lending Club default with profit-based decisions
- **Decision:** Predict P2P loan default and fund loans based on expected
  profit, not just a classification score.
- **Why:** The user prefers the finance domain. This option was the easiest
  of the three finance ideas to start with. The profit angle and the
  leakage-safe approach set it apart from typical Lending Club notebooks.
- **Rejected:** Freddie Mac mortgage delinquency (stronger drift story, more
  data engineering); HMDA mortgage approvals with a fairness audit (more
  distinctive, messier fields).

### D-002: Dataset source: Kaggle mirror
- **Decision:** `wordsforthewise/lending-club` →
  `accepted_2007_to_2018Q4.csv.gz` (about 2.26M loans × about 151 columns).
- **Why:** Lending Club removed its official downloads in 2020. This mirror
  is complete and widely used.
- **Risk / mitigation:** It isn't an official source, so we record the
  file's checksum with DVC and document where it came from in the README.

### D-003: Target definition
- **Decision:** `1` = Charged Off / Default, `0` = Fully Paid. Current, late
  and grace-period loans are **dropped**.
- **Why:** Their final outcome isn't known yet. Labeling them "not default"
  would be wrong.

### D-004: Handle maturity bias: 36-month loans issued ≤ 2015
- **Decision:** Keep 36-month loans issued on or before 2015-12. Leave out
  60-month loans in v1.
- **Why:** The data is a snapshot from the end of 2018. Recent loans can
  only appear "finished" if they ended early (usually by defaulting or being
  prepaid), which distorts the default rate. Only loans with enough time to
  finish give honest labels. For 60-month loans, only issue years up to 2013
  would qualify, which is too little data.
- **Revisit if:** we want a v2 with a separate model for 60-month loans.

### D-005: Time-based train/validation/test split
- **Decision:** Train on 2007-06 → 2013-12, validate on 2014, test on 2015.
  Loans from 2016 onward are used only for simulated drift monitoring.
- **Why:** In real use the model always predicts *future* loans. A random
  split would mix future information into training, and the scores would
  look better than they really are.

### D-006: Feature allowlist, not blocklist
- **Decision:** Only columns on an explicit allowlist (in `params.yaml`) can
  become features. Everything else is dropped by default.
- **Why:** With about 150 columns, a blocklist would eventually miss a
  column that leaks the answer. An allowlist fails safe: a new or unfamiliar
  column is left out unless we deliberately add it.

### D-007: `int_rate` and `grade` are allowed features
- **Decision:** Keep them.
- **Why:** Lending Club assigns them *before* the listing is shown, so an
  investor sees them. This means our model has to beat Lending Club's own
  risk grading, which is the honest test for an investor.

### D-008: Post-issuance columns only for evaluation
- **Decision:** `total_pymnt`, `recoveries` and similar columns are never
  features. They are used only to compute each loan's **realized profit**
  when evaluating.
- **Why:** They describe what happened *after* funding. As features they
  would leak the answer. For measuring actual profit, they're exactly the
  data we need.

### D-009: Decide by expected profit
- **Decision:** Fund a loan if
  `(1 − p) × gain_if_paid − p × LGD × principal > 0`. The loss given
  default (LGD) is estimated from **training data only**. The final
  threshold is tuned on validation data to maximize realized profit.
- **Why:** The investor's goal is money, not accuracy. The two kinds of
  mistake cost different amounts, and the amounts differ from loan to loan.

### D-010: Baselines to beat
- **Decision:** (1) Fund every loan, (2) fund only grades A–B,
  (3) logistic regression.
- **Why:** A model only proves its value if it does better than simple rules
  an investor could use without any ML.

### D-011: Build round 1 scope: Phases 0–4
- **Decision:** Setup, data + EDA, preprocessing pipeline, CV + model
  comparison, and evaluation + profit policy, plus tests, README and
  `requirements.txt`. The serving API, orchestration and monitoring come in
  later rounds.
- **Why:** Small rounds are easier to understand and review. The production
  layers are added on top of a model that already works.

### D-012: Same MLOps stack as Hotel Booking roadmap
- **Decision:** uv, DVC, MLflow, Pandera, FastAPI, Docker, Prefect, GitHub
  Actions, Evidently.
- **Why:** Using the same tools again on a different problem helps them
  stick, and gives the portfolio a consistent story.

### D-013: EDA in a notebook, pipeline in scripts
- **Decision:** `notebooks/01_eda.ipynb` for exploration, with outputs kept.
  All logic the model depends on lives in `src/`.
- **Why:** Notebooks are good for visual, back-and-forth exploration, and
  GitHub displays their charts. Scripts can be tested, reviewed in Git and
  rerun the same way every time. Findings from the notebook become code or
  configuration, never logic that lives only in a notebook.

### D-014: `uv` + `pyproject.toml` with exported `requirements.txt`
- **Decision:** `pyproject.toml` + `uv.lock` define the dependencies.
  `requirements.txt` is generated with `uv export`.
- **Why:** There is one source of exact versions, and anyone using plain
  `pip install -r requirements.txt` still gets the same versions.

### D-015: Polars for data, pandas at the sklearn boundary
- **Decision:** Polars (lazy scan, reading only needed columns) → Parquet
  for loading and cleaning. Convert to pandas only when handing data to
  scikit-learn or SHAP.
- **Why:** The raw CSV is several GB and the machine has 16 GB of RAM.
  Polars uses much less memory and is faster. scikit-learn and SHAP expect
  pandas.

### D-016: Two-layer preprocessing
- **Decision:**
  - **Layer A (row-by-row, run once on all data):** parse strings, derive
    ratios and credit-history length, create labels, filter rows.
  - **Layer B (learns from data, inside an sklearn `Pipeline`):**
    imputation, outlier caps, encoders, scaling. Fitted on training
    folds only.
- **Why:** This is the main way we prevent leakage. A step that learns a
  statistic (a median, a category list, a mean) leaks information if it
  ever sees validation or test rows. Putting those steps in a `Pipeline`
  means cross-validation refits them on each fold automatically.
  Column-drop decisions are also made from training data only.

### D-017: Missing values handled by why they're missing
- **Decision:**
  - "Never happened" (e.g. `mths_since_last_delinq`) → a placeholder value
    (999) plus a missing flag
  - Missing at random (`revol_util`, `dti`) → the training median plus a
    missing flag
  - Fields that didn't exist during training years (`open_acc_6m`,
    `il_util`, …) → dropped
  - `emp_length = n/a` → its own category, `unknown`
- **Why:** Missing values mean different things in different columns, and
  the fact that a value is missing can itself help predict default.

### D-018: Categorical encoding
- **Decision:** `grade`/`sub_grade` as ordered numbers (A1 < … < G5).
  One-hot encoding for `home_ownership`, `verification_status`, `purpose`
  and `addr_state`. Levels under 1% of training rows are grouped into
  `other`, and unseen levels are handled safely at prediction time. No
  target encoding in v1.
- **Why:** Grades have a meaningful order that one-hot encoding would throw
  away. Grouping rare levels keeps unusual values from breaking the model.
  Target encoding is easy to get wrong in a way that leaks.

### D-019: Numeric transforms
- **Decision:** Apply `log1p` to `annual_inc` and `revol_bal`, and cap them
  at the training 99.5th percentile. Standard scaling for logistic
  regression only.
- **Why:** These amounts are heavily skewed with extreme outliers, which
  distort a linear model. Tree models don't need scaling.

### D-020: Keep correlated features
- **Decision:** Keep correlated pairs such as `int_rate`/`sub_grade` and
  `loan_amnt`/`installment`.
- **Why:** LightGBM isn't hurt by correlated features, and L2
  regularization keeps logistic regression stable.
- **Trade-off:** SHAP explanations are harder to interpret when features
  are correlated. We'll note this in the README.

### D-021: No resampling for class imbalance
- **Decision:** No SMOTE, oversampling or undersampling. Handle the
  imbalance through metrics (PR-AUC, log loss), a threshold chosen by
  profit, and `class_weight` tried as a tuning option (with
  recalibration if it's used).
- **Why:** The profit rule multiplies by the predicted probability, so the
  probabilities must be honest. Resampling changes how often the model sees
  defaults, which shifts all its predictions. SMOTE also invents
  unrealistic synthetic borrowers from mixed data. With about 15–20%
  defaults, the imbalance is moderate, not extreme.

### D-022: Expanding-window cross-validation by year
- **Decision:** 3 folds: train on 2007–10 → validate on 2011; 2007–11 →
  2012; 2007–12 → 2013. Built as a custom splitter by calendar year.
- **Why:** Each fold only ever trains on the past. sklearn's
  `TimeSeriesSplit` splits by row count, not by year.

### D-023: Algorithms: logistic regression vs. LightGBM
- **Decision:** Compare regularized logistic regression with LightGBM.
- **Why:** Logistic regression is easy to explain and is the standard
  approach in credit risk. LightGBM captures non-linear patterns and
  interactions and is strong on tabular data. Comparing them shows whether
  the extra complexity is worth it.

### D-024: Tuning, calibration, and threshold procedure
- **Decision:** Tune hyperparameters with CV → refit on 2007–2013 → fit
  the calibrator on 2014 → choose the profit threshold on 2014 → score 2015
  **once**. The model is not refit on 2007–2014.
- **Why:** The 2014 data is needed to fit the calibrator and choose the
  threshold. Training on it too would make both of them optimistic. The
  test set is scored only once so the reported result stays honest.

### D-025: Metrics
- **Decision:**
  - Tuning: **log loss**
  - Reported: ROC-AUC, PR-AUC, Brier score, reliability curve
  - **Final business metric:** realized return on the 2015 test set (total
    $, return per invested $, % of loans funded, default rate among funded
    loans) compared with the baselines
- **Why:** The decision depends on the probability value, and log loss
  rewards probabilities that are both accurate and well calibrated.
  Accuracy is misleading: predicting "no default" for everything scores
  about 83%. F1 treats both kinds of mistake as equally costly, but here a
  missed default costs about 3× more than a wrongly rejected loan.

### D-026: "Does not meet credit policy" loans kept, with a flag
- **Decision:** Include them, mapped to 0 or 1 by their status, with a
  flag column.
- **Why:** They are real, finished loans. The flag lets the model treat
  them differently.
- **Revisit if:** EDA in Phase 1 shows they behave very differently or only
  appear in years that distort the training data.

### D-027: Out of scope for v1
- **Decision:** Leave out the rejected-loans file, NLP on `desc` and
  `emp_title`, cloud deployment, Kubernetes, Terraform, and portfolio
  optimization under a budget.
- **Why:** These are good ideas for later rounds. Leaving them out keeps v1
  focused on a model that is correct and free of leakage.

### D-028: Raw data via manual download, stored as uncompressed CSV
- **Decision:** The dataset was downloaded by hand from Kaggle, not with the
  Kaggle CLI, and extracted to `data/raw/accepted_2007_to_2018Q4.csv`
  (1,675,133,810 bytes; 2,260,702 lines including the header).
  SHA-256: `3eae03c28fd9d2e8a076ebeb73507e8d4d0f44d90500decdb0936e0933d1f36a`.
  The file will be tracked with DVC, not Git.
- **Why:** Kaggle API credentials aren't set up, and a manual download works
  just as well for a one-time pull. The CSV is kept uncompressed because
  Polars can only read the needed columns from a plain CSV; it can't do that
  with a `.gz`. The checksum lets anyone confirm they have exactly the same
  file (the source isn't official, see D-002).
- **Revisit if:** we automate data download in the Phase 6 retraining flow;
  that needs the Kaggle CLI and credentials.

---

## Template for new decisions

```markdown
### D-0XX: <short title>
- **Decision:** <what we chose>
- **Why:** <the reasoning>
- **Rejected:** <alternatives considered> (optional)
- **Revisit if:** <condition that would change this> (optional)
```

Add a row to the **Index** table as well.
