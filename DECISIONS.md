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
| [D-040](#d-040-prune-redundant-features-86---73) | Prune redundant features: 86 -> 73 | Features | Accepted | 2026-09-23 |
| [D-041](#d-041-missing-indicators-are-built-unconditionally) | Missing indicators built unconditionally | Features | Accepted | 2026-09-23 |
| [D-042](#d-042-every-config-key-must-do-something) | Every config key must do something (or be deleted) | Structure | Accepted | 2026-09-23 |
| [D-043](#d-043-audit-confirmations-and-known-limitations) | Audit confirmations and known limitations | Evaluation | Accepted | 2026-09-23 |
| [D-044](#d-044-early-stopping-inside-folds-median-trees-for-the-final-fit) | Early stopping inside folds; median tree count for the final fit | Modeling | Accepted | 2026-09-23 |
| [D-045](#d-045-the-probabilistic-baseline-is-lending-clubs-own-grade) | Probabilistic baseline = Lending Club's own grade | Evaluation | Accepted | 2026-09-23 |
| [D-046](#d-046-class_weightbalanced-rejected-on-evidence) | `class_weight="balanced"` rejected on evidence | Modeling | Accepted | 2026-09-23 |
| [D-047](#d-047-mlflow-tracks-to-sqlite-not-the-mlruns-folder) | MLflow tracks to SQLite, not the `mlruns/` folder | Tooling | Accepted | 2026-09-23 |
| [D-048](#d-048-compare-policies-at-equal-selectivity) | Compare policies at equal selectivity | Evaluation | Accepted | 2026-09-23 |
| [D-049](#d-049-calibrate-on-2014-h1-tune-the-policy-on-2014-h2) | Calibrate on 2014 H1, tune the policy on 2014 H2 | Validation | Accepted | 2026-09-23 |
| [D-050](#d-050-the-expected-value-rule-funds-almost-everything) | Report the expected-value rule even though it funds ~everything | Business | Accepted | 2026-09-23 |
| [D-051](#d-051-calibration-helps-brier-but-not-log-loss-here) | Calibration helps Brier but not log loss here; keep it, report both | Evaluation | Accepted | 2026-09-23 |
| [D-052](#d-052-profit-economics-are-dollar-weighted) | Profit economics are dollar-weighted, not per-loan averages | Business | Accepted | 2026-09-23 |
| [D-053](#d-053-ranking-policies-use-the-raw-score-not-the-calibrated-probability) | Ranking policies use the raw score, not the calibrated probability | Evaluation | Accepted | 2026-09-23 |
| [D-054](#d-054-every-headline-comparison-carries-a-confidence-interval) | Every headline comparison carries a bootstrap confidence interval | Evaluation | Accepted | 2026-09-23 |
| [D-055](#d-055-the-threshold-is-an-interior-optimum-not-a-constraint-artifact) | The chosen threshold is an interior optimum | Validation | Accepted | 2026-09-23 |
| [D-056](#d-056-training-and-serving-share-one-derivation-function) | Training and serving share one derivation function | Serving | Accepted | 2026-09-23 |
| [D-057](#d-057-ship-one-bundle-not-four-artifacts) | Ship one bundle (model + calibrator + threshold + economics) | Serving | Accepted | 2026-09-23 |
| [D-058](#d-058-never-pickle-from-__main__) | Never pickle classes defined in `__main__` | Serving | Accepted | 2026-09-23 |
| [D-059](#d-059-serving-dependencies-are-separate-from-training-ones) | Serving dependencies separate from training ones | Tooling | Accepted | 2026-09-23 |
| [D-060](#d-060-a-promotion-gate-judged-on-money) | Retraining has a promotion gate, judged on money | Modeling | Accepted | 2026-09-23 |
| [D-061](#d-061-the-gate-never-looks-at-the-test-year) | The gate never looks at the test year | Validation | Accepted | 2026-09-23 |
| [D-062](#d-062-ci-runs-the-whole-pipeline-on-synthetic-loans) | CI runs the whole pipeline on synthetic loans | Structure | Accepted | 2026-09-23 |
| [D-063](#d-063-the-demo-bundle-refuses-to-overwrite-a-real-one) | The demo bundle refuses to overwrite a real one | Serving | Accepted | 2026-09-23 |
| [D-064](#d-064-2016-2018-loans-are-replayed-as-monthly-production-batches) | 2016-2018 loans are replayed as monthly production batches | Monitoring | Accepted | 2026-09-23 |
| [D-065](#d-065-the-labels-we-can-see-early-are-not-a-sample-we-can-score-on) | The labels we can see early are not a sample we can score on | Monitoring | Accepted | 2026-09-23 |
| [D-066](#d-066-the-drift-test-is-pinned-and-run-on-raw-inputs) | The drift test is pinned, and run on raw inputs | Monitoring | Accepted | 2026-09-23 |
| [D-067](#d-067-the-retraining-trigger-needs-persistence-not-a-spike) | The retraining trigger needs persistence, not a spike | Monitoring | Accepted | 2026-09-23 |
| [D-068](#d-068-drift-and-breakage-are-different-alerts) | Drift and breakage are different alerts | Monitoring | Accepted | 2026-09-23 |
| [D-069](#d-069-the-html-report-is-written-for-three-months-not-thirty-six) | The HTML report is written for three months, not thirty-six | Monitoring | Accepted | 2026-09-23 |
| [D-070](#d-070-documentation-that-can-be-re-run) | Documentation that can be re-run, not prose | Docs | Accepted | 2026-09-23 |
| [D-071](#d-071-addr_state-is-a-proxy-that-earns-nothing) | `addr_state` is a proxy that earns nothing | Fairness | **Open** | 2026-09-23 |

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

### D-040: Prune redundant features (86 -> 73)
- **Decision:** Three changes after a correlation audit of the training
  features:
  1. **Missing indicators only where missingness predicts default.** Adding one
     for every numeric column produced **seven perfectly correlated columns**,
     because the same ~30 loans from 2007 are missing every bureau field.
     Indicators are kept only for `emp_length_years` (4.3% missing, 18.2%
     default vs. 12.4%), `pub_rec_bankruptcies` (0.8% missing, 24.5% vs.
     12.5%) and `revol_util` (0.1% missing, 16.9% vs. 12.6%).
  2. **No indicator for the structural columns.** The 999 sentinel already
     says "never happened"; the flag was perfectly correlated with the value.
  3. **`drop="if_binary"` for one-hot encoding.** A two-level column such as
     `initial_list_status` produced two perfectly collinear columns.
- **Why:** Duplicated columns add no information. They inflate the input,
  slow training, and split the importance of one real effect across several
  columns, which makes SHAP explanations misleading.
- **Kept deliberately:** `loan_amnt`/`installment` (0.99), `grade`/`sub_grade`
  (0.97), `int_rate`/`sub_grade` (0.96), `loan_to_income`/
  `installment_to_income` (0.99). These are genuinely different measurements
  that happen to move together, and both model types handle them (D-020).

### D-041: Missing indicators are built unconditionally
- **Decision:** Use `MissingIndicator(features="all")` instead of
  `SimpleImputer(add_indicator=True)`.
- **Why:** `add_indicator=True` creates an indicator **only for columns that
  actually contained a missing value while fitting**. A cross-validation fold
  (or a production batch) with no missing `revol_util` would produce one
  feature fewer, and a model trained on 73 features cannot score 72. This is
  the same class of bug as D-036, found by a test that used clean data.

### D-042: Every config key must do something
- **Decision:** An audit compared `params.yaml` against the code and found
  four keys that described behavior nobody had implemented. All are now real:
  - `data.drop_statuses` — the prepare stage now **raises** if the data
    contains a `loan_status` value that is in none of the three lists, so a
    refreshed dataset cannot silently drop loans.
  - `data.keep_credit_policy_loans` — the switch now actually filters.
  - `features.max_missing_fraction` — a guard now checks it on the **training
    split only**, exempting the structural columns (D-031).
  - `features.never_happened_columns` and `features.log_columns` were
    **deleted**: they duplicated `features.groups`, and two copies of the same
    list drift apart.
- **Why:** Configuration that does nothing is worse than no configuration.
  Someone (including a future you) changes the value, sees no effect, and
  loses trust in the whole file.

### D-043: Audit confirmations and known limitations
- **Confirmed by measurement:**
  1. **The 2015 cutoff is safe.** Only 0.05% of 36-month loans issued in 2015
     were still unresolved at the snapshot (147 of 283,173), versus 28% for
     2016 and 60% for 2017. The test set is not distorted by survivorship.
  2. **`total_pymnt` already includes post-charge-off recoveries** — it equals
     principal + interest + late fees + recoveries to within $1 for 100% of
     charged-off loans. So LGD = 0.365 is right, and recoveries must **not**
     be added again.
  3. **No loan appears in two splits** (checked on seven identifying columns).
  4. No zero-variance or duplicated feature columns, and no non-finite values.
- **Known limitations, to state in the README:**
  - Profit ignores Lending Club's ~1% service fee on payments and the
    collection fee on recoveries. Including the collection fee alone would
    raise LGD from 0.365 to 0.377, so our profit numbers are slightly
    optimistic in absolute terms; the comparison between strategies is
    unaffected because every strategy is measured the same way.
  - Profit ignores the time value of money: $1 repaid in month 1 counts the
    same as $1 in month 36.
  - `initial_list_status` drifts sharply (82% "f" in training, 52% in
    validation) because Lending Club changed how loans were listed. It is
    known at listing time, so it stays, but it is a platform artifact rather
    than a borrower trait.

### D-044: Early stopping inside folds, median trees for the final fit
- **Decision:** During cross-validation, each fold's validation set decides
  when LightGBM stops adding trees. The final model, trained on all of
  2007-2013, uses the **median stopping point across the three folds**
  (325, 195, 275 -> 275 trees).
- **Why:** Early stopping needs a held-out set, and there is none once we
  train on everything. Taking the median means no single fold dictates the
  final model.
- **Caveat, stated honestly:** the fold scores are slightly optimistic,
  because the stopping point was chosen on the same rows that produced the
  score. The numbers that matter for the final claim come from the 2015 test
  set in Phase 4, which nothing has touched.

### D-045: The probabilistic baseline is Lending Club's own grade
- **Decision:** `GradePriorClassifier` predicts the historical default rate of
  the loan's grade, measured on the training rows of each fold. The fund-all
  and grade A-B rules stay as **policy** baselines for Phase 4.
- **Why:** Phase 3 compares probabilities, and "fund everything" produces no
  probability. The grade prior is the honest zero-effort benchmark: the grade
  is assigned before the listing appears, so an investor gets it for free.
- **Result:** grade alone scores log loss 0.36015 / ROC-AUC 0.6232. Logistic
  regression reaches 0.35345 / 0.6618 and LightGBM 0.35172 / 0.6672, so the
  extra columns do add information beyond Lending Club's grading — modestly.

### D-046: `class_weight="balanced"` rejected on evidence
- **Decision:** Keep the natural class balance. `class_weight="balanced"` was
  included in the search and lost decisively.
- **Why:** The numbers are a textbook illustration of D-021. With C=0.01:
  | setting | ROC-AUC | log loss | Brier |
  |---|---|---|---|
  | `class_weight=None` | 0.6618 | **0.35345** | **0.10331** |
  | `class_weight="balanced"` | 0.6626 | 0.63249 | 0.22138 |

  Ranking quality is unchanged (ROC-AUC is the same to three decimals), but
  log loss and Brier score nearly double: re-weighting makes the model behave
  as if half of all loans default, so every probability is inflated. For a
  decision rule that multiplies by the probability, that is fatal — which is
  exactly why the plan ruled out resampling from the start.

### D-047: MLflow tracks to SQLite, not the `mlruns/` folder
- **Decision:** `tracking_uri: sqlite:///mlflow.db`.
- **Why:** MLflow 3 put the file-based store into maintenance mode and refuses
  to use it without an opt-out flag. A database backend is also what a real
  tracking server uses, so this is closer to production anyway. The database
  file is git-ignored; `reports/cv_results.json` is the committed, reviewable
  record of the search.

### D-048: Compare policies at equal selectivity
- **Decision:** Besides the tuned threshold, every model is also scored at the
  **same share of loans funded as the grade A-B rule**, with that share
  measured on the 2014 tuning rows, never on test.
- **Why:** A pickier policy usually shows a higher return per dollar simply
  because it skips more loans. Without matching selectivity, "our model beats
  the grade rule" can just mean "our model funds fewer loans".
- **Result on the 2015 test year (return per dollar):**
  | strategy | share funded | return per $ |
  |---|---|---|
  | fund everything | 100% | +0.0655 |
  | grade A-B rule | 57.2% | +0.0718 |
  | logistic regression, matched | 59.5% | +0.0732 |
  | **LightGBM, matched** | **56.1%** | **+0.0739** |
  | LightGBM, tuned threshold (p < 0.15) | 66.9% | +0.0734 |

  So the model genuinely beats Lending Club's grading at equal selectivity,
  and its tuned policy earns nearly as much per dollar while deploying
  **10 percentage points more capital**.

### D-049: Calibrate on 2014 H1, tune the policy on 2014 H2
- **Decision:** Split the validation year in two: fit the isotonic calibrator
  on January-June 2014 (71,955 loans), choose the funding threshold on
  July-December 2014 (90,615 loans).
- **Why:** Choosing the threshold on the same rows that fitted the calibrator
  would make both look better than they are. Splitting costs nothing here,
  since both halves are large.
- **Note:** the test year is scored exactly once, after both are frozen.

### D-050: The expected-value rule funds almost everything
- **Decision:** Keep reporting the "fund if expected value > 0" rule, even
  though it funds 99.9% of loans and therefore matches the fund-all baseline.
- **Why:** This is a real finding, not a bug. With a measured loss given
  default of 0.373 and a prepayment factor of 0.829, a typical loan breaks
  even at a default probability far above what any loan is predicted to have.
  At Lending Club's interest rates, almost every loan is worth funding in
  expectation.
- **Consequence:** the model's value is **not** in rejecting loans with
  negative expected value; it is in **ranking** when capital is limited. An
  investor with $1B cannot fund the whole 2015 book ($3.62B), so which 30-60%
  they choose is the entire game. That is why the threshold policy, not the
  expected-value rule, is the headline.
- **In dollars, per $1B deployed:** fund-all $65.5M, grade A-B $71.8M,
  logistic regression $72.1M, **LightGBM $73.4M** — about +12% more profit
  than funding everything, and +2.2% more than the grade rule.

### D-051: Calibration helps Brier but not log loss here
- **Decision:** Keep the isotonic calibration, and report the metrics before
  and after rather than only the flattering one.
- **What happened:** on the 2015 test year, LightGBM's Brier score improved
  (0.12099 -> 0.12064) while its log loss got slightly worse
  (0.39752 -> 0.39924).
- **Why:** the calibrator learned the relationship on 2014 H1, where 13.22% of
  loans defaulted. The test year defaults at 14.89%. Calibration therefore
  pulls probabilities toward a world that is slightly safer than the one being
  scored. This is prior drift, the same effect the EDA found across splits.
- **Why it does not change the funding decision:** isotonic regression is
  **monotone**, so it never changes the ranking of loans. Any policy of the
  form "fund the safest X%" is unaffected. Calibration matters for the
  expected-value rule and for anyone reading a probability as a probability.
- **Revisit if:** a later phase adds drift-triggered recalibration, which is
  exactly the production answer to this problem (Phase 7).

### D-052: Profit economics are dollar-weighted
- **Decision:** Loss given default and the prepayment factor are computed as
  total dollars lost / total dollars lent, not as the average of per-loan
  ratios. LGD changes from 0.3732 to **0.3657**.
- **Why:** A portfolio's return is decided by dollars. Averaging per-loan
  ratios gives a $1,000 loan the same weight as a $35,000 one. The gap between
  the two figures exists because smaller loans default somewhat more often, so
  the per-loan average overstates the loss an investor actually takes.
- **Also fixed:** `min_share_funded` and `bootstrap_draws` moved from
  hard-coded defaults into `params.yaml` — the same rule as D-042.

### D-053: Ranking policies use the raw score, not the calibrated probability
- **Decision:** "Fund when the score is below t" uses the model's raw output.
  Calibrated probabilities are still used for the expected-value rule and for
  reporting.
- **Why:** Isotonic regression is a **step function**. It maps the 283,026 test
  loans onto only **255 distinct values**, and the largest single value covers
  **39,792 loans (14% of the portfolio)**. A cut-off landing inside that block
  moves 14% of the capital in or out at once, so a tiny change in data could
  swing the policy wildly. The raw score is continuous (283,026 distinct
  values), so the policy degrades smoothly.
- **Effect on the result:** essentially none (test return per dollar 0.0735 vs.
  0.0734), which is the point: the fix costs nothing and removes a fragility.
- **Trade-off:** the threshold is no longer readable as "fund if risk < 15%".
  The calibrated probability is still reported for that interpretation.

### D-054: Every headline comparison carries a confidence interval
- **Decision:** Bootstrap the test loans 1,000 times and report a 95% interval
  for each difference in return per dollar.
- **Why:** The gaps are small — 0.0021 per dollar between the model and the
  grade rule. A point estimate cannot tell whether that is skill or luck on
  one particular year's loans.
- **Results (all on the 2015 test year):**
  | comparison | difference | 95% CI | P(better) |
  |---|---|---|---|
  | LightGBM threshold vs. fund everything | +0.0080 | [+0.0072, +0.0087] | 1.000 |
  | LightGBM matched vs. grade A-B | +0.0021 | [+0.0015, +0.0027] | 1.000 |
  | LightGBM threshold vs. grade A-B | +0.0017 | [+0.0011, +0.0024] | 1.000 |
  | logistic regression matched vs. grade A-B | +0.0015 | [+0.0008, +0.0022] | 1.000 |
  | LightGBM vs. logistic regression (matched) | +0.0006 | [+0.0001, +0.0011] | 0.994 |

  Every interval excludes zero, so the edge is real — including LightGBM's
  small advantage over logistic regression.
- **Caveat that stays in the README:** this measures sampling noise within one
  test year. It says nothing about how the model would hold up in a different
  credit cycle, which is a larger risk than sampling error.

### D-055: The threshold is an interior optimum, not a constraint artifact
- **Checked:** the profit curve on the tuning half of 2014 rises from +0.0719
  at a 2.7% funding share to a peak of **+0.0879 around a score of 0.14-0.155
  (about 63% funded)**, then falls back to +0.0825 when funding everything.
- **Why it matters:** if the best threshold had sat at the `min_share_funded`
  boundary, the policy would have been decided by an arbitrary constraint
  rather than by the data. The top five thresholds cluster in 0.135-0.16, so
  the choice is also stable rather than a spike on one lucky grid point.

### D-056: Training and serving share one derivation function
- **Decision:** The API accepts **raw listing fields** only, and derives
  `fico_mid`, `credit_history_months`, `loan_to_income`,
  `installment_to_income`, `emp_length_years` and `term_months` itself, using
  the same `derive_features()` that the training pipeline calls.
- **Why:** Six of the 30 model inputs are engineered. Asking callers to supply
  them would guarantee training/serving skew eventually: someone computes
  credit history in months from a different date, and the model quietly scores
  nonsense while still returning plausible numbers. Sharing one function makes
  the skew impossible rather than merely unlikely.
- **Verified by test:** 25 real 2015 loans scored through the HTTP API match
  the offline pipeline's probabilities to 1e-6, and every funding decision
  agrees.

### D-057: Ship one bundle, not four artifacts
- **Decision:** `LoanDecisionModel` packages the pipeline, the calibrator, the
  threshold and the profit economics (LGD, prepayment factor) into a single
  file, with metadata: training window, CV score, test return, creation time.
- **Why:** A decision needs four things fitted at four different stages. Ship
  them separately and production drifts out of sync — someone retrains the
  model and leaves the old threshold in place, and nothing fails loudly. The
  bundle means the service either has a complete, consistent decision or it
  has nothing.
- **Also:** `/health` reports which model version is loaded, because a deploy
  check needs to know *which* model is live, not just that something answers.

### D-058: Never pickle classes defined in `__main__`
- **Decision:** The bundle is built through `serving/__main__.py`, which
  imports the class from `serving/bundle.py`. `bundle.py` has no
  `if __name__ == "__main__"` block.
- **Why:** Found by a failing test. Running `python -m lending_club.serving.bundle`
  defines `LoanDecisionModel` in the module `__main__`, and joblib records that
  name inside the artifact. Any other process then fails with
  *"Can't get attribute 'LoanDecisionModel' on module '__main__'"*. The file
  looked fine and the artifact was written successfully — it simply could not
  be loaded anywhere else, which is exactly the kind of bug that surfaces in
  production rather than in development.

### D-059: Serving dependencies are separate from training ones
- **Decision:** `pyproject.toml` keeps only runtime packages in
  `dependencies`; MLflow, DVC, SHAP, matplotlib, seaborn and pandera moved to
  a `train` extra. Two lock exports: `requirements.txt` (everything, 940
  packages) and `requirements-serve.txt` (serving only, **104**).
- **Why:** The container cannot use an experiment tracker or a plotting
  library. Every package shipped is extra image size, extra build time and
  extra attack surface. The split also documents which code is production code.
- **Container details:** multi-stage build, non-root user (`appuser`),
  `libgomp1` for LightGBM's OpenMP runtime (absent from `python:slim`), and a
  `HEALTHCHECK` that calls `/health`. Image size: **1.03 GB**, build time ~10s
  after the dependency layer is cached.
- **Verified end to end:** 200 real 2015 loans were scored through the running
  container and compared with the offline pipeline. Maximum probability
  difference **4.9e-07** (API rounding to six decimals), maximum expected-return
  difference **$0.005** (rounding to cents), and **every funding decision
  identical**. That is the Phase 5 exit criterion met.

### D-060: A promotion gate, judged on money
- **Decision:** The Prefect flow ends with a champion/challenger gate. A newly
  trained bundle replaces the live one only if its funding policy earns more
  **return per dollar** on validation data, by more than a 0.0005 margin.
- **Why judged on money:** a model can improve its log loss and still pick a
  worse portfolio. The gate should test the thing the system is for.
- **Why a margin:** without one, production would be swapped for a 0.00001
  improvement — noise — and every swap carries real risk.
- **Demonstrated:** running the flow on unchanged data produced
  "REJECTED - candidate +0.08803 vs champion +0.08803 (+0.00000, below the
  0.0005 margin)". A scheduled retrain with no gate is a way to ship a worse
  model automatically, with every task green.

### D-061: The gate never looks at the test year
- **Decision:** The promotion gate scores candidates on the policy-tuning half
  of 2014. The flow still recomputes the 2015 test metrics, but they are
  logged as "report only" and no decision uses them.
- **Why:** the test year is scored once, on purpose. A gate that consulted it
  on every retrain would turn it into another tuning set — each run picking
  whatever happens to score best there — and the headline result would slowly
  stop meaning anything.
- **Production note:** in a real deployment the fixed 2015 test year would be
  replaced by a rolling recent window, with the gate still judging on data the
  candidate has never seen.

### D-062: CI runs the whole pipeline on synthetic loans
- **Decision:** `lending_club.testing.synthetic` generates loans whose default
  risk genuinely depends on grade, FICO and loan-to-income. The end-to-end test
  runs the real code over them: clean -> label -> split -> preprocess -> train
  -> calibrate -> choose a policy -> bundle -> serve over HTTP, and asserts the
  model reaches AUC > 0.60 on the planted signal.
- **Why:** GitHub runners have no access to the 1.6 GB dataset, and "every unit
  passes" is not "the pipeline works". This catches wiring mistakes — a renamed
  column, a stage that no longer matches the next one — that unit tests cannot
  see. It runs in under two seconds.
- **CI jobs:** lint (`ruff check` + `format --check`), tests, then a container
  job that builds a demo bundle, builds the image, starts it, and asserts
  `/health` and `/predict` return a valid decision.

### D-063: The demo bundle refuses to overwrite a real one
- **Decision:** `build_demo_bundle()` raises `FileExistsError` if a bundle is
  already present, unless `--force` is passed.
- **Why:** found the hard way. Running the demo builder locally silently
  replaced the real trained bundle with one fitted on synthetic data. The API
  kept working and kept returning confident answers — from a toy model. CI
  starts from a clean checkout, so the guard never fires there, and locally it
  prevents a failure mode that would be very hard to notice.

### D-064: 2016-2018 loans are replayed as monthly production batches
- **Decision:** the 988,585 36-month loans issued 2016-01 to 2018-12 are scored
  by the **deployed bundle** month by month and treated as production traffic.
  `production_frame()` shares `derive_features` with training but deliberately
  does not call `prepare.transform()`.
- **Why:** `transform()` drops every loan without a final status, which is the
  entire population we want to monitor. Scoring with the bundle (not the bare
  model) means monitoring measures what the service actually deploys, including
  the calibrator and the funding threshold.
- **Result:** 36 monthly batches. Feature drift is immediate and large -
  between 43% and 60% of model inputs are flagged in every single month.

### D-065: The labels we can see early are not a sample we can score on
- **Decision:** the share of loans with a known outcome is reported as
  *coverage* and never used as a performance metric.
- **Why:** a 36-month loan issued in 2018-06 can only have a final status by
  the 2018Q4 snapshot if it ended early - paid off ahead of schedule, or
  charged off in the first months. Coverage falls from 99% (2016-01) to 3%
  (2018-12), and the observed default rate on that visible subset swings from
  14.3% to 20.5% and back to 12.6% **purely from label maturity**, not because
  the world changed. Charting it would look exactly like a model going bad.
- **This is the reason drift monitoring exists.** The outcome of today's
  decision arrives in three years; waiting for it is not a strategy. So we
  watch the two things visible immediately: the inputs and the model's output.

### D-066: The drift test is pinned, and run on raw inputs
- **Decision:** Wasserstein for numeric columns, Jensen-Shannon for
  categorical, threshold 0.15, applied to the 30 raw model inputs.
- **Why (pinned):** left alone, Evidently chooses the statistical test based on
  sample size - a distance test for large samples, a p-value test for small
  ones. A 45,000-loan month and a 900-loan month would then be measured with
  different tests, in opposite directions (high = drift vs. low = drift), and
  the monthly series would not be comparable. Pinning makes every month one
  bounded distance where higher always means more drift.
- **Why (raw inputs):** a transformed column is partly an artifact of the
  fitted preprocessor, and "feature 47 drifted" is not actionable. "Income
  verification mix changed" is. A test asserts the monitored set equals the
  model's input set, so a feature can never be added without being watched.

### D-067: The retraining trigger needs persistence, not a spike
- **Decision:** retrain when the drifted share exceeds 0.30 **or** prediction
  drift exceeds 0.15, for 2 consecutive months. One alert per episode; the run
  resets on a clean month.
- **Why:** a single month is often an artifact - a marketing push, a holiday, a
  product change that reverts. Requiring persistence trades a month of delay
  for far fewer false alarms, and retraining is not free: it consumes the
  promotion gate (D-060) and every model swap carries risk.
- **Why one alert per episode:** the condition here persists for 36 months. An
  alert that fires every month is a log line, not an alert. The suppression
  assumes the first alert is acted on - after a retrain the reference window
  moves and the clock restarts.

### D-068: Drift and breakage are different alerts
- **Decision:** missing-value jumps and never-before-seen categories raise an
  `investigate` alert, not `retrain`. They fire immediately, without waiting
  for persistence.
- **Why:** retraining fixes a world that has moved. It does not fix a column
  that suddenly arrives empty - and firing "retrain" at a broken feed would
  train the next model on the same broken data. Different problem, different
  owner, different urgency.
- **Found in production data:** `addr_state=ND` (Lending Club did not lend in
  North Dakota during the training window) and `home_ownership=ANY` (a category
  introduced in 2016-07). Neither breaks the service - the one-hot encoder maps
  unseen levels to "infrequent" (D-018) - but both are worth knowing about, and
  both are handled as knowledge, reported once rather than every month.

### D-069: The HTML report is written for three months, not thirty-six
- **Decision:** `drift.json` records every month; Evidently's HTML report is
  saved only for the months listed in `monitoring.html_months`, plus a single
  committed PNG summarising all 36.
- **Why:** each HTML file embeds the plotting library and the data, at ~5.6 MB.
  Thirty-six of them came to 201 MB of artifact nobody would open. The JSON is
  the machine-readable record; the HTML is what a human opens when something
  fires, and any month can be regenerated on demand.

### D-070: Documentation that can be re-run
- **Decision:** the model card is a separate `MODEL_CARD.md`, not a README
  section; the architecture diagram is Mermaid in the README, not a PNG; and the
  fairness audit is a module (`lending_club.fairness`) with a DVC stage and
  tests, not a paragraph of prose.
- **Why:** documentation rots because nothing checks it. A Mermaid diagram is
  diffable in a pull request and cannot silently disagree with a refactor the
  way an exported image does. A fairness section that *asserts* "addr_state may
  be a proxy" is box-ticking; one that regenerates `reports/fairness.json` from
  the deployed bundle can be re-run when the model changes and will contradict
  itself if the claim stops being true.
- **Also fixed here:** the README told a new reader to run
  `uv sync --extra dev`, which installs neither DVC, MLflow nor Evidently —
  `dvc repro`, the one command the whole project rests on, would have failed
  immediately. Verified with `uv sync --extra dev --dry-run`: it would have
  uninstalled 185 packages. The Phase 8 exit criterion is that someone new can
  clone and reproduce, so this counted as a real bug, not a typo.
- **Rejected:** a `docs/` site (Sphinx/MkDocs). For a portfolio project the
  audience reads the repository on GitHub; a build step between them and the
  content is a cost with no benefit.

### D-071: `addr_state` is a proxy that earns nothing
- **Status: OPEN.** Recommended, deliberately not yet applied.
- **Finding:** `addr_state` is a well-documented proxy for race and income in
  the US. On the 2015 test year it produces:
  - a funding-rate spread of **47.8% (NV) to 74.0% (MA)**, against 66.3% overall;
  - a realized default rate among *funded* loans of **5.8% (OR) to 14.3% (AR)**,
    against 10.0% overall — the model is not equally right about each state;
  - a correlation between mean score and actual default across states of only
    **+0.51**, versus **+0.94** for `purpose`. `purpose` spreads far wider
    (14.2% to 82.0%) and is almost entirely earned; `addr_state` is not.
- **And it contributes nothing.** 1.6% of total SHAP weight. Refitting the
  selected LightGBM configuration on the training split without it:

  | | log loss (2014) | ROC-AUC |
  |---|---|---|
  | with `addr_state` | 0.37692 | 0.6769 |
  | without | 0.37693 | **0.6770** |

  A difference of 0.00001, with AUC marginally *better* without the feature.
- **Recommendation:** remove `addr_state` from `features.allowlist` and from
  `features.groups.nominal`, then `dvc repro`. A feature that is a recognised
  proxy, produces a 26-point funding spread and a 2.5x spread in realized error,
  and earns no measurable accuracy, does not belong in a credit model.
- **Why it is still open:** it is a modelling change, not a documentation one.
  It invalidates every published number, the bundle and the monitoring baseline,
  so it is the owner's call rather than something to slip into a docs phase.
- **Revisit if:** the model is ever used for anything resembling an approval
  decision — at that point this stops being a recommendation.

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
