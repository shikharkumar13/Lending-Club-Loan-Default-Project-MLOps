"""Phase 3: compare models with expanding-window cross-validation.

Three things make this honest rather than merely convenient:

1. **The preprocessor is refitted inside every fold.** Medians, caps and
   category lists come only from that fold's training rows (D-016).
2. **Folds respect time.** Each fold trains on earlier years and validates on
   the next one (D-022), the way the model would really be used.
3. **Tuning uses log loss.** The funding rule multiplies by the predicted
   probability, so a probability that is confidently wrong must be punished —
   ranking alone (ROC-AUC) is not enough (D-025).

Everything is logged to MLflow: one parent run per model family, one child run
per configuration, so the whole search stays comparable and reproducible.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from lending_club.config import load_params, path_of
from lending_club.data.split import cv_folds
from lending_club.features.pipeline import build_preprocessor, to_model_frame
from lending_club.models.baselines import GradePriorClassifier

Folds = list[tuple[np.ndarray, np.ndarray]]


def score(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Metrics chosen for an imbalanced problem where probabilities matter (D-025)."""
    return {
        # Optimized: rewards probabilities that are both accurate and calibrated.
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        # Ranking quality, insensitive to the class balance.
        "roc_auc": float(roc_auc_score(y_true, proba)),
        # Ranking quality for the minority class specifically.
        "pr_auc": float(average_precision_score(y_true, proba)),
        # Squared error of the probabilities: a calibration check.
        "brier": float(brier_score_loss(y_true, proba)),
    }


def run_cv(
    make_estimator: Callable[[], Any],
    kind: str,
    X: pd.DataFrame,
    y: np.ndarray,
    folds: Folds,
    params: dict,
    early_stop: bool = False,
    preprocess: bool = True,
) -> dict[str, Any]:
    """Fit one configuration across all folds and average the scores.

    `preprocess=False` passes the raw columns through: the grade baseline reads
    the `grade` column directly and needs no encoding.
    """
    per_fold, best_iters = [], []
    patience = params["model"]["lightgbm"]["early_stopping_rounds"]

    for train_idx, val_idx in folds:
        if preprocess:
            # The preprocessor is fitted on THIS fold's training rows only.
            preprocessor = build_preprocessor(params, kind)
            X_train = preprocessor.fit_transform(X.iloc[train_idx])
            X_val = preprocessor.transform(X.iloc[val_idx])
        else:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        estimator = make_estimator()
        if early_stop:
            # The fold's validation set decides when to stop adding trees. This
            # is standard practice, and slightly optimistic: the stopping point
            # has seen that fold. The final model uses the median stopping point
            # across folds, so no single fold dictates the result (D-044).
            estimator.fit(
                X_train,
                y_train,
                eval_X=X_val,
                eval_y=y_val,
                eval_metric="binary_logloss",
                callbacks=[early_stopping(patience, verbose=False)],
            )
            best_iters.append(int(estimator.best_iteration_ or 0))
        else:
            estimator.fit(X_train, y_train)

        per_fold.append(score(y_val, estimator.predict_proba(X_val)[:, 1]))

    mean = {k: float(np.mean([f[k] for f in per_fold])) for k in per_fold[0]}
    std = {f"{k}_std": float(np.std([f[k] for f in per_fold])) for k in per_fold[0]}
    return {"mean": mean, "std": std, "per_fold": per_fold, "best_iters": best_iters}


def sample_configs(search: dict[str, list], n_iter: int, seed: int) -> list[dict]:
    """Randomized search: n_iter random combinations, not the full grid.

    A full grid over 7 settings would be thousands of fits. Random sampling
    finds nearly as good a configuration for a tiny fraction of the compute,
    because only a few of the settings matter much.
    """
    rng = np.random.default_rng(seed)
    seen, configs = set(), []
    while len(configs) < n_iter:
        config = {key: values[rng.integers(len(values))] for key, values in search.items()}
        key = tuple(sorted((k, str(v)) for k, v in config.items()))
        if key not in seen:
            seen.add(key)
            configs.append({k: (v.item() if hasattr(v, "item") else v) for k, v in config.items()})
    return configs


def paired_comparison(candidates: list[dict]) -> dict[str, Any]:
    """Compare the best of each family FOLD BY FOLD (D-079).

    Averages hide the thing that matters. Here the gap between LightGBM and
    logistic regression (~0.0017 log loss) is an order of magnitude smaller than
    the spread between folds (~0.026), because 2011, 2012 and 2013 are genuinely
    different years. That spread is common to both models, so the honest test is
    paired: does the same model win in every fold, and by how much relative to
    the variation in that difference?

    A winner that takes 3 folds out of 3 by a consistent margin is a result. A
    winner that takes 2 of 3 is a coin toss, and should be reported as one.
    """
    best = {}
    for family in ("grade_prior", "logistic_regression", "lightgbm"):
        family_candidates = [c for c in candidates if c["family"] == family]
        if family_candidates:
            best[family] = min(family_candidates, key=lambda c: c["mean"]["log_loss"])

    per_fold = {
        family: [fold["log_loss"] for fold in candidate["per_fold"]]
        for family, candidate in best.items()
    }
    pairs = [
        ("lightgbm", "logistic_regression"),
        ("lightgbm", "grade_prior"),
        ("logistic_regression", "grade_prior"),
    ]
    comparisons = {}
    for a, b in pairs:
        if a not in per_fold or b not in per_fold:
            continue
        # Lower log loss is better, so b - a > 0 means `a` won that fold.
        differences = [y - x for x, y in zip(per_fold[a], per_fold[b], strict=True)]
        wins = sum(d > 0 for d in differences)
        comparisons[f"{a}_vs_{b}"] = {
            "per_fold_difference": [round(d, 5) for d in differences],
            "folds_won": wins,
            "folds": len(differences),
            "mean_difference": round(float(np.mean(differences)), 5),
            "std_difference": round(float(np.std(differences, ddof=1)), 5),
            "consistent": wins == len(differences),
        }
    return {"per_fold_log_loss": per_fold, "pairs": comparisons}


def train() -> dict[str, Any]:
    params = load_params()
    seed = params["seed"]

    train_frame = pl.read_parquet(path_of("processed_dir") / "train.parquet")
    X = to_model_frame(train_frame, params)
    y = train_frame["target"].to_numpy()
    folds = cv_folds(train_frame, params)
    print(f"train: {len(X):,} loans, {X.shape[1]} input columns, {len(folds)} folds")

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment"])

    candidates: list[dict[str, Any]] = []

    # --- baseline: Lending Club's own grade ---------------------------------
    with mlflow.start_run(run_name="grade_prior"):
        started = time.time()
        result = run_cv(GradePriorClassifier, "tree", X, y, folds, params, preprocess=False)
        mlflow.log_params({"model": "grade_prior"})
        mlflow.log_metrics(result["mean"] | result["std"])
        candidates.append({"family": "grade_prior", "config": {}, **result})
        print(
            f"grade_prior            log_loss={result['mean']['log_loss']:.5f} "
            f"roc_auc={result['mean']['roc_auc']:.4f} ({time.time() - started:.0f}s)"
        )

    # --- logistic regression -------------------------------------------------
    lr_cfg = params["model"]["logistic_regression"]
    with mlflow.start_run(run_name="logistic_regression"):
        for C in lr_cfg["grid"]["C"]:
            for class_weight in lr_cfg["grid"]["class_weight"]:
                config = {"C": C, "class_weight": class_weight}
                with mlflow.start_run(run_name=f"lr_C{C}_{class_weight}", nested=True):
                    result = run_cv(
                        lambda c=config: LogisticRegression(
                            **lr_cfg["fixed"], **c, random_state=seed
                        ),
                        "linear",
                        X,
                        y,
                        folds,
                        params,
                    )
                    mlflow.log_params({"model": "logistic_regression"} | config)
                    mlflow.log_metrics(result["mean"] | result["std"])
                candidates.append({"family": "logistic_regression", "config": config, **result})
                print(
                    f"logreg C={C:<5} cw={str(class_weight):<8} "
                    f"log_loss={result['mean']['log_loss']:.5f} "
                    f"roc_auc={result['mean']['roc_auc']:.4f}"
                )

    # --- lightgbm ------------------------------------------------------------
    lgb_cfg = params["model"]["lightgbm"]
    configs = sample_configs(lgb_cfg["search"], lgb_cfg["n_iter"], seed)
    with mlflow.start_run(run_name="lightgbm"):
        for i, config in enumerate(configs, start=1):
            started = time.time()
            with mlflow.start_run(run_name=f"lgbm_{i:02d}", nested=True):
                result = run_cv(
                    lambda c=config: LGBMClassifier(**lgb_cfg["fixed"], **c, random_state=seed),
                    "tree",
                    X,
                    y,
                    folds,
                    params,
                    early_stop=True,
                )
                mlflow.log_params({"model": "lightgbm"} | config)
                mlflow.log_metrics(result["mean"] | result["std"])
                mlflow.log_metric("median_best_iteration", float(np.median(result["best_iters"])))
            candidates.append({"family": "lightgbm", "config": config, **result})
            print(
                f"lgbm {i:02d}/{len(configs)}  log_loss={result['mean']['log_loss']:.5f} "
                f"roc_auc={result['mean']['roc_auc']:.4f} "
                f"trees={int(np.median(result['best_iters']))} ({time.time() - started:.0f}s)"
            )

    # --- pick the best of each family and refit on the full training split ---
    models_dir = path_of("models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for family in ("grade_prior", "logistic_regression", "lightgbm"):
        best = min(
            (c for c in candidates if c["family"] == family), key=lambda c: c["mean"]["log_loss"]
        )
        kind = "linear" if family == "logistic_regression" else "tree"

        if family == "grade_prior":
            estimator = GradePriorClassifier()
            # No preprocessing: it reads the raw `grade` column.
            model = Pipeline([("model", estimator)])
            model.fit(X, y)
            joblib.dump(model, models_dir / f"{family}.joblib")
            summary.append(
                {"family": family, "config": {}, "cv": best["mean"], "cv_std": best["std"]}
            )
            print(f"saved {family}: cv log_loss={best['mean']['log_loss']:.5f}")
            continue
        elif family == "logistic_regression":
            estimator = LogisticRegression(**lr_cfg["fixed"], **best["config"], random_state=seed)
        else:
            # Trees are no longer chosen by early stopping (there is no held-out
            # set once we train on everything), so use the median stopping point
            # from cross-validation.
            n_estimators = max(int(np.median(best["best_iters"])), 50)
            estimator = LGBMClassifier(
                **{**lgb_cfg["fixed"], "n_estimators": n_estimators},
                **best["config"],
                random_state=seed,
            )
            best["config"] = best["config"] | {"n_estimators": n_estimators}

        model = Pipeline([("preprocess", build_preprocessor(params, kind)), ("model", estimator)])
        model.fit(X, y)
        joblib.dump(model, models_dir / f"{family}.joblib")
        summary.append(
            {"family": family, "config": best["config"], "cv": best["mean"], "cv_std": best["std"]}
        )
        print(f"saved {family}: cv log_loss={best['mean']['log_loss']:.5f}")

    reports = path_of("reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    comparison = paired_comparison(candidates)
    payload = {
        # per_fold is KEPT (D-079). Averages alone cannot tell you whether a
        # winner won consistently or won one fold by luck, and stripping them
        # made that impossible to check after the fact.
        "candidates": candidates,
        "selected": summary,
        "comparison": comparison,
        "search_budget": {
            "grade_prior": 1,
            "logistic_regression": len(lr_cfg["grid"]["C"]) * len(lr_cfg["grid"]["class_weight"]),
            "lightgbm": len(configs),
        },
    }
    # Trailing newline so the end-of-file pre-commit hook does not rewrite the
    # file after every training run.
    (reports / "cv_results.json").write_text(json.dumps(payload, indent=2) + "\n")

    print("\npaired comparison across folds (positive = first model wins):")
    for name, stats in comparison["pairs"].items():
        verdict = "consistent" if stats["consistent"] else "NOT consistent"
        print(
            f"  {name:42s} {stats['mean_difference']:+.5f} "
            f"+/- {stats['std_difference']:.5f}  "
            f"{stats['folds_won']}/{stats['folds']} folds  {verdict}"
        )
    return {"selected": summary, "comparison": comparison}


if __name__ == "__main__":
    train()
