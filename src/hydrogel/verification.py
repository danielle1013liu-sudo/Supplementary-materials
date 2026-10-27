"""Compare a reproduced run with its numerical reference tables."""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT

KEYS = ["target", "family", "imputer", "model_seed", "scope"]


def _check_rows(actual: pd.DataFrame, reference: pd.DataFrame,
                keys: list[str], values: list[str], atol: float) -> float:
    if actual.empty or actual.duplicated(keys).any():
        raise ValueError("A result table is empty or contains duplicate row identities.")
    if reference.duplicated(keys).any():
        raise ValueError("The reference table contains duplicate row identities.")
    configurations = actual[["target", "family", "imputer"]].drop_duplicates()
    reference = reference.merge(configurations, on=["target", "family", "imputer"], how="inner")
    left = actual.set_index(keys)
    right = reference.set_index(keys)
    if len(left) != len(right) or set(left.index) != set(right.index):
        raise ValueError("Result identities do not match the complete reference populations and realisations.")
    right = right.reindex(left.index)
    if right[values].isna().any().any():
        raise ValueError("Some result identities are absent from the reference table.")
    observed = left[values].to_numpy(dtype=float)
    expected = right[values].to_numpy(dtype=float)
    if not np.isfinite(observed).all():
        raise ValueError("A result table contains non-finite values.")
    np.testing.assert_allclose(observed, expected, rtol=0, atol=atol)
    return float(np.max(np.abs(observed - expected)))


def verify_run(run: Path, root: Path | None = None, atol: float = 1e-7) -> dict:
    """Verify every row written by a comparison or P32 case run."""
    root = Path(root) if root is not None else ROOT
    predictions = pd.read_csv(run / "predictions.csv", float_precision="round_trip")
    metrics = pd.read_csv(run / "metrics.csv", float_precision="round_trip")
    prediction_groups = set(predictions[KEYS].itertuples(index=False, name=None))
    metric_groups = set(metrics[KEYS].itertuples(index=False, name=None))
    if prediction_groups != metric_groups:
        raise ValueError("Prediction and metric tables describe different configurations or scopes.")
    is_p32 = set(predictions["scope"]) == {"P32"}
    filename = "p32_predictions.csv" if is_p32 else "predictions.csv.gz"
    reference = pd.read_csv(root / "reference" / filename, float_precision="round_trip")
    prediction_error = _check_rows(
        predictions, reference, KEYS + ["record_id"],
        ["observed_pg_ml", "predicted_pg_ml"], atol,
    )
    if is_p32:
        from .evaluation import regression_metrics

        rows = []
        for group, frame in reference.groupby(KEYS, sort=False):
            rows.append({**dict(zip(KEYS, group)), **regression_metrics(
                frame["observed_pg_ml"], frame["predicted_pg_ml"],
            )})
        reference_metrics = pd.DataFrame(rows)
    else:
        reference_metrics = pd.read_csv(
            root / "reference" / "metrics.csv", float_precision="round_trip",
        )
    metric_error = _check_rows(
        metrics, reference_metrics, KEYS, ["R2", "RMSE_pg_ml", "MAE_pg_ml"], atol,
    )
    return {
        "verified": True,
        "prediction_rows": len(predictions),
        "metric_rows": len(metrics),
        "max_prediction_difference": prediction_error,
        "max_metric_difference": metric_error,
        "absolute_tolerance": atol,
    }

