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
| [D-029](#d-029-phase-0-tooling-choices) | Phase 0 tooling: ruff + pre-commit, params.yaml as single config, DVC pointers in git | Tooling | Accepted | 2026-09-23 |
| [D-030](#d-030-drop-four-columns-after-the-phase-1-audit) | Drop `mort_acc`, `tot_cur_bal`, `tot_coll_amt`, `application_type` | Features | Accepted | 2026-09-23 |
| [D-031](#d-031-structurally-missing-columns-are-exempt-from-the-drop-rule) | "Never happened" columns exempt from the missingness drop rule | Features | Accepted | 2026-09-23 |
| [D-032](#d-032-loss-given-default-is-365-not-65) | Loss given default is 36.5%, measured, not 65% | Business | Accepted | 2026-09-23 |
| [D-033](#d-033-column-parsing-matches-this-mirrors-formats) | Parsing matches this mirror's formats; `dti` 999 and zero income become missing | Data | Accepted | 2026-09-23 |
| [D-034](#d-034-pure-transform-functions-separate-from-file-io) | Pure transform functions, separate from file I/O | Structure | Accepted | 2026-09-23 |
| [D-035](#d-035-feature-groups-live-in-paramsyaml) | Feature groups in `params.yaml`; drop `term_months`, raw FICO columns, `issue_date` | Features | Accepted | 2026-09-23 |
| [D-036](#d-036-imputers-keep-empty-features) | Imputers use `keep_empty_features=True` | Features | Accepted | 2026-09-23 |
| [D-037](#d-037-schema-validation-runs-inside-the-split-stage) | Schema validation runs inside the split stage | Structure | Accepted | 2026-09-23 |
| [D-038](#d-038-dvc-pipeline-with-three-stages) | DVC pipeline: ingest -> prepare -> split | Tooling | Accepted | 2026-09-23 |
| [D-039](#d-039-keep-three-cv-folds-despite-the-small-first-fold) | Keep 3 CV folds, equally weighted, despite a small first fold | Validation | Accepted | 2026-09-23 |

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
- **Phase 1 result:** 2,438 such loans survive the filters (0.4% of the data),
  they default at 26.6% vs. 13.95% overall, and they appear only in
  2007–2010. Kept with the flag as planned. The flag is always `false` in the
  validation and test years, so it only helps the model interpret the early
  training loans.

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

### D-029: Phase 0 tooling choices
- **Decision:**
  - **`params.yaml` is the single source of configuration.** Split dates, the
    feature allowlist, forbidden column prefixes, and model settings all live
    there, not in the code.
  - **The `.dvc` pointer files are committed to git** (`.gitignore` uses
    `data/**` plus exceptions), while the data files themselves are not.
  - **ruff + pre-commit** run automatically on every commit: lint, format,
    trailing whitespace, and a block on files over 5 MB.
  - **`dvc config core.autostage true`** so DVC pointer changes are staged
    automatically.
  - **hatchling** packaging with a `src/` layout, so `lending_club` is
    importable everywhere without `sys.path` hacks.
- **Why:** Configuration in one file means `dvc repro` can tell when a
  parameter changed and rerun only what's affected, and every experiment can
  be traced back to an exact config. Committing the pointer files is what
  links a git commit to the exact data version it used. The large-file hook
  is a safety net so a stray 1.6 GB CSV can never enter git history, where it
  would be permanent.
- **Note:** LightGBM 4.7 runs on this Mac without `brew install libomp`; the
  wheel bundles OpenMP. DVC's cache uses APFS copy-on-write, so tracking the
  1.6 GB file cost about 1 GB of disk rather than a full second copy.

### D-030: Drop four columns after the Phase 1 audit
- **Decision:** Remove `mort_acc`, `tot_cur_bal`, `tot_coll_amt` and
  `application_type` from the feature allowlist.
- **Why:** The first three are **100% missing for every loan issued before
  2012** — Lending Club only started reporting them later. In the training
  years they are 21–31% empty, and whether a value is present says "this is an
  old loan", not anything about the borrower. A model would learn that, and it
  would be useless in production, where every loan has the field. Imputing
  them would quietly invent a value for a third of the training data.
  `application_type` is 99.96% "Individual" in our window (239 joint
  applications out of 621,022), so it carries no signal.
- **Rejected:** Keeping them with a missing flag (the flag is really a
  year indicator); starting training in 2012 instead (that would cut the
  training data to two years and leave too few cross-validation folds).
- **Revisit if:** a v2 restricts the project to 2013+ loans, where all three
  columns are fully populated.

### D-031: Structurally missing columns are exempt from the drop rule
- **Decision:** `mths_since_last_delinq` (58% missing) and
  `mths_since_last_record` (90% missing) are kept, despite the
  `max_missing_fraction: 0.5` rule.
- **Why:** Here missing means "this borrower has never had a delinquency or a
  public record", which is a *good* sign, not absent data. The information is
  preserved as a sentinel value (999) plus a missing flag. The drop rule is
  meant for columns that are empty by accident, not by meaning.

### D-032: Loss given default is 36.5%, not 65%
- **Decision:** Charged-off loans repay on average **63.5%** of the funded
  amount, so the loss given default is **0.365**. Realized profit is
  `total_pymnt - funded_amnt`.
- **Why:** Measured directly from the data, rather than assumed. Defaulting
  borrowers usually make many payments before they stop, and some money is
  recovered afterwards, so a default is nothing like a total loss.
- **Consequence:** Rejecting a loan is *more* costly than we assumed, so the
  profit-maximizing threshold will be higher (more loans funded) than a plan
  based on a 65% loss would suggest.
- **Also measured:** repaid loans return about +15.5% over three years, and
  the average loan earns **+$1,029** (+8.25% per dollar invested). So
  "fund everything" is a profitable baseline, and the model has to beat a
  positive number, not zero.
- **Note:** This figure is measured on all matured loans. Phase 4 re-estimates
  it on the training split only, so no test information reaches the policy.

### D-033: Column parsing matches this mirror's formats
- **Decision:** In this Kaggle mirror, `int_rate` and `revol_util` are
  **already numeric**, so no percent-string parsing is needed. Parsing is only
  required for `term` (" 36 months"), `emp_length` ("10+ years"), and the
  dates `issue_d` / `earliest_cr_line` ("Dec-2015"). `dti >= 999` (a
  "not available" sentinel, 5 loans) and `annual_inc = 0` (2 loans) are
  converted to missing.
- **Why:** The plan assumed the raw Lending Club string formats. Checking the
  actual file avoided writing parsing code that would have silently produced
  all-null columns.

### D-034: Pure transform functions, separate from file I/O
- **Decision:** Each pipeline stage splits into a pure function (`transform`)
  and a thin wrapper that reads and writes files (`prepare`).
- **Why:** Tests can then feed the logic a handful of handmade rows instead of
  the 1.6 GB dataset. The 12 tests covering labeling, filters and derived
  features run in under a tenth of a second, so they can run on every commit.

### D-035: Feature groups live in params.yaml
- **Decision:** `features.groups` in `params.yaml` lists which columns get
  which treatment (skewed amount, numeric, structurally missing, ordinal,
  nominal, boolean). The pipeline code reads those lists. Three columns are
  deliberately excluded from the model:
  - `term_months` — always 36 in v1, so it carries no information
  - `fico_range_low` / `fico_range_high` — replaced by their midpoint,
    `fico_mid`; keeping all three would add two near-identical columns
  - `issue_date` — used for splitting only. As a feature it would encode
    "which year this loan came from", which cannot transfer to a new loan.
- **Why:** Changing the pipeline then means editing configuration, not code,
  and DVC can see that a parameter changed and rerun the affected stage.
- **Result:** 30 input columns become 86 model features after encoding.

### D-036: Imputers keep empty features
- **Decision:** All imputers are built with `keep_empty_features=True`.
- **Why:** Found while testing. By default, scikit-learn **silently drops** a
  column that is entirely missing in the data it was fitted on. In one
  cross-validation fold that would change the number of features, and a model
  trained that way would break when served a batch where the column is
  present. Keeping empty features guarantees the same output shape every time.

### D-037: Schema validation runs inside the split stage
- **Decision:** A Pandera schema checks each processed split (types, value
  ranges, no forbidden columns) before it is written to disk.
- **Why:** It caught three real problems the moment it was added: `target`
  was `Int8` rather than `Int64`, `term_months` was `Int16`, and `annual_inc`
  has 4 missing values (all from 2007) that the plan assumed could not exist.
  Without the schema, those would have surfaced much later as a confusing
  model error.
- **How it applies:** `annual_inc` is now declared nullable and the 4 loans are
  imputed by the pipeline, like any other missing value.

### D-038: DVC pipeline with three stages
- **Decision:** `dvc.yaml` defines `ingest` -> `prepare` -> `split`, with each
  stage declaring its code, data and parameter dependencies.
- **Why:** `dvc repro` reruns only what actually changed. Editing a split date
  in `params.yaml` reruns the split stage but not the two-minute CSV read.
  It also means nobody has to remember the running order, and the rebuild is
  identical on another machine.

### D-039: Keep three CV folds despite the small first fold
- **Decision:** Keep the folds as planned: train 2007-10 -> validate 2011
  (17k/14k rows), 2007-11 -> 2012 (32k/43k), 2007-12 -> 2013 (75k/100k). Each
  fold counts equally in the average score.
- **Why:** The early Lending Club years are genuinely small: 2007-2010 is only
  17,433 loans out of 175,426 in training. Dropping the first fold would leave
  two folds and lose the oldest data; weighting folds by size would let 2013
  dominate the tuning, which is closer to the test period but gives a less
  robust estimate.
- **Revisit if:** Phase 3 scores swing widely between folds, which would mean
  the first fold is adding noise rather than information.

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
