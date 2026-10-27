"""Fixed-configuration evaluation with fold-local preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .data import load_c4, load_config, load_dataset, load_schema, load_split
from .models import make_estimator
from .preprocessing import make_preprocessor

IDENTIFIERS = ["target", "family", "imputer", "model_seed", "scope"]
METRICS = ["R2", "RMSE_pg_ml", "MAE_pg_ml"]
PREDICTION_COLUMNS = IDENTIFIERS + [
    "record_id", "observed_pg_ml", "predicted_pg_ml"
]


@dataclass
class EvaluationResult:
    """Per-realisation results and the corresponding fitted pipelines."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    models: dict[int, Pipeline]


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate metrics on the original concentration scale."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.ndim != 1 or observed.shape != predicted.shape or len(observed) < 2:
        raise ValueError("Metrics require matching one-dimensional arrays of at least two observations.")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Observed and predicted concentrations must be finite.")
    return {
        "R2": float(r2_score(observed, predicted)),
        "RMSE_pg_ml": float(np.sqrt(mean_squared_error(observed, predicted))),
        "MAE_pg_ml": float(mean_absolute_error(observed, predicted)),
    }


def _append_scope(
    metric_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    identity: dict[str, Any],
    scope: str,
    record_ids: list[str],
    observed: np.ndarray,
    predicted: np.ndarray,
) -> None:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(record_ids) != len(observed):
        raise ValueError(f"Record identities do not match the {scope} observations.")
    context = {**identity, "scope": scope}
    metric_rows.append({**context, **regression_metrics(observed, predicted)})
    prediction_rows.extend(
        {
            **context,
            "record_id": record_id,
            "observed_pg_ml": float(value),
            "predicted_pg_ml": float(estimate),
        }
        for record_id, value, estimate in zip(record_ids, observed, predicted, strict=True)
    )


def _partition(
    frame: pd.DataFrame, membership: dict[str, Any], n_splits: int
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    train_ids = list(membership["train_ids"])
    test_ids = list(membership["test_ids"])
    if not frame.record_id.is_unique:
        raise ValueError("Dataset record identities must be unique.")
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("A partition contains duplicate record identities.")
    if set(train_ids) & set(test_ids):
        raise ValueError("Training and Test memberships overlap.")
    if set(train_ids) | set(test_ids) != set(frame.record_id):
        raise ValueError("Training and Test memberships must partition the complete dataset.")
    labels = np.asarray(membership["fold_labels"])
    if labels.shape != (len(train_ids),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Fold labels must be integers aligned with the ordered Training records.")
    if set(labels.tolist()) != set(range(1, n_splits + 1)):
        raise ValueError(f"Fold labels must contain each value from 1 to {n_splits}.")
    indexed = frame.set_index("record_id", drop=False)
    return indexed.loc[train_ids].copy(), indexed.loc[test_ids].copy(), labels


def evaluate_configuration(
    target: str,
    family: str,
    imputer: str | None = None,
    root: str | Path | None = None,
) -> EvaluationResult:
    """Evaluate one fixed model configuration using the saved partition.

    Validation metrics use pooled out-of-fold Training predictions. Each
    preprocessor is fitted once on its own fitting population and reused across
    the three model realisations. The primary fitted estimator predicts Training,
    Test and C4 together, in that order, from one transformed input batch.
    """
    target = target.lower()
    configuration = load_config(target, root=root)
    schema = load_schema(root=root)
    frame = load_dataset(target, root=root)
    membership = load_split(target, root=root)
    cases = load_c4(target, root=root)
    if family not in configuration["models"]:
        raise ValueError(f"No fixed configuration is available for {family!r}.")
    specification = configuration["models"][family]
    method = specification["imputer"] if imputer is None else imputer
    if method not in configuration["imputation_methods"]:
        raise ValueError(f"Unsupported imputation method: {method!r}.")
    seeds = configuration["model_seeds"]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("The model comparison requires three distinct model seeds.")
    training, test, labels = _partition(frame, membership, configuration["n_splits"])
    if set(cases.record_id) & set(frame.record_id):
        raise ValueError("C4 identities must be distinct from the curated dataset identities.")

    features = list(schema["predictors"])
    x_train = training[features]
    y_train = training.cytokine_pg_ml.to_numpy(dtype=float)
    preprocessor = make_preprocessor(schema, method)
    transformed_training = preprocessor.fit_transform(x_train, y_train)
    joint = pd.concat([x_train, test[features], cases[features]], ignore_index=True)
    transformed_joint = preprocessor.transform(joint)
    if not np.isfinite(transformed_training).all() or not np.isfinite(transformed_joint).all():
        raise ValueError("Preprocessing produced non-finite descriptor values.")

    folds = []
    for label in range(1, configuration["n_splits"] + 1):
        fit_index = np.flatnonzero(labels != label)
        valid_index = np.flatnonzero(labels == label)
        fold_preprocessor = make_preprocessor(schema, method)
        x_fit = fold_preprocessor.fit_transform(x_train.iloc[fit_index], y_train[fit_index])
        x_valid = fold_preprocessor.transform(x_train.iloc[valid_index])
        if not np.isfinite(x_fit).all() or not np.isfinite(x_valid).all():
            raise ValueError(f"Preprocessing produced non-finite values in fold {label}.")
        folds.append((fit_index, valid_index, x_fit, x_valid))

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fitted: dict[int, Pipeline] = {}
    n_train, n_test = len(training), len(test)
    for seed in seeds:
        model = make_estimator(family, specification["parameters"], seed)
        model.fit(transformed_training, y_train)
        prediction = np.asarray(model.predict(transformed_joint), dtype=float)
        if prediction.shape != (len(joint),):
            raise ValueError("The fitted model returned an unexpected prediction shape.")
        oof = np.full(n_train, np.nan)
        for fit_index, valid_index, x_fit, x_valid in folds:
            fold_model = make_estimator(family, specification["parameters"], seed)
            fold_model.fit(x_fit, y_train[fit_index])
            oof[valid_index] = fold_model.predict(x_valid)
        identity = {"target": target, "family": family, "imputer": method, "model_seed": int(seed)}
        scopes = [
            ("Training", training.record_id.tolist(), y_train, prediction[:n_train]),
            ("Validation", training.record_id.tolist(), y_train, oof),
            ("Test", test.record_id.tolist(), test.cytokine_pg_ml.to_numpy(dtype=float),
             prediction[n_train:n_train + n_test]),
            ("C4", cases.record_id.tolist(), cases.observed_pg_ml.to_numpy(dtype=float),
             prediction[n_train + n_test:]),
        ]
        for scope, ids, observed, estimates in scopes:
            _append_scope(metric_rows, prediction_rows, identity, scope, ids, observed, estimates)
        fitted[int(seed)] = Pipeline([("preprocess", preprocessor), ("model", model)])
    return EvaluationResult(
        pd.DataFrame(metric_rows, columns=IDENTIFIERS + METRICS),
        pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS),
        fitted,
    )


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarise individual model metrics as mean and sample SD (ddof=1)."""
    missing = set(IDENTIFIERS + METRICS) - set(metrics.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}.")
    keys = ["target", "family", "imputer", "scope"]
    if metrics.duplicated(IDENTIFIERS).any():
        raise ValueError("Each model realisation must contribute one metric row per scope.")
    if not np.isfinite(metrics[METRICS].to_numpy(dtype=float)).all():
        raise ValueError("Cannot summarise non-finite metrics.")
    rows = []
    for values, group in metrics.groupby(keys, sort=False, dropna=False):
        if len(group) != 3:
            raise ValueError("Each summary requires three individual model realisations.")
        row = dict(zip(keys, values, strict=True))
        row["n_models"] = len(group)
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_SD"] = float(group[metric].std(ddof=1))
        rows.append(row)
    columns = keys + ["n_models"] + [field for metric in METRICS for field in (f"{metric}_mean", f"{metric}_SD")]
    return pd.DataFrame(rows, columns=columns)
