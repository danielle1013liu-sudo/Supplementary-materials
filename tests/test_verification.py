import pandas as pd
import pytest

from hydrogel.data import ROOT
from hydrogel.evaluation import regression_metrics, summarize_metrics
from hydrogel.verification import verify_run


@pytest.fixture
def reference_run(tmp_path):
    for filename in ["predictions.csv.gz", "metrics.csv"]:
        frame = pd.read_csv(ROOT / "reference" / filename, float_precision="round_trip")
        frame = frame.loc[(frame.target == "tnf") & (frame.family == "RF") & (frame.imputer == "median")]
        frame.to_csv(tmp_path / filename.removesuffix(".gz"), index=False, float_format="%.17g")
    summarize_metrics(frame).to_csv(tmp_path / "summary.csv", index=False, float_format="%.17g")
    return tmp_path


def test_reference_comparison_checks_all_rows(reference_run):
    report = verify_run(reference_run)
    assert report["verified"]
    assert report["prediction_rows"] == 3 * (120 + 120 + 30 + 2)
    assert report["max_prediction_difference"] == 0
    assert report["summary_rows"] == 4
    assert report["max_summary_difference"] == 0


def test_missing_model_prediction_is_rejected(reference_run):
    path = reference_run / "predictions.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.iloc[:-1].to_csv(path, index=False, float_format="%.17g")
    with pytest.raises(ValueError, match="complete reference"):
        verify_run(reference_run)


def test_modified_prediction_is_rejected(reference_run):
    path = reference_run / "predictions.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[0, "predicted_pg_ml"] += 1
    frame.to_csv(path, index=False, float_format="%.17g")
    with pytest.raises(AssertionError):
        verify_run(reference_run)


def test_different_model_metrics_are_rejected(reference_run):
    frame = pd.read_csv(ROOT / "reference" / "metrics.csv", float_precision="round_trip")
    frame = frame.loc[(frame.target == "tnf") & (frame.family == "LGBM") & (frame.imputer == "median")]
    frame.to_csv(reference_run / "metrics.csv", index=False, float_format="%.17g")
    with pytest.raises(ValueError, match="different configurations or scopes"):
        verify_run(reference_run)


@pytest.mark.parametrize("column", [
    "R2_mean", "R2_SD", "RMSE_pg_ml_mean", "RMSE_pg_ml_SD", "MAE_pg_ml_mean", "MAE_pg_ml_SD",
])
def test_modified_summary_value_is_rejected(reference_run, column):
    path = reference_run / "summary.csv"
    summary = pd.read_csv(path, float_precision="round_trip")
    summary.loc[0, column] += 0.01
    summary.to_csv(path, index=False, float_format="%.17g")
    with pytest.raises(AssertionError):
        verify_run(reference_run)


def test_population_sd_in_summary_is_rejected(reference_run):
    path = reference_run / "summary.csv"
    summary = pd.read_csv(path, float_precision="round_trip")
    metrics = pd.read_csv(reference_run / "metrics.csv", float_precision="round_trip")
    group = metrics.loc[metrics.scope.eq(summary.iloc[0].scope)]
    summary.loc[0, "R2_SD"] = group.R2.std(ddof=0)
    summary.to_csv(path, index=False, float_format="%.17g")
    with pytest.raises(AssertionError):
        verify_run(reference_run)


@pytest.mark.parametrize("damage", ["row", "identity", "duplicate", "count", "nonfinite", "column"])
def test_incomplete_or_invalid_summary_is_rejected(reference_run, damage):
    path = reference_run / "summary.csv"
    summary = pd.read_csv(path, float_precision="round_trip")
    if damage == "row":
        summary = summary.iloc[:-1]
    elif damage == "identity":
        summary.loc[0, "imputer"] = "unknown"
    elif damage == "duplicate":
        summary = pd.concat([summary, summary.iloc[[0]]], ignore_index=True)
    elif damage == "count":
        summary.loc[0, "n_models"] = 2
    elif damage == "nonfinite":
        summary.loc[0, "R2_mean"] = float("inf")
    else:
        summary = summary.drop(columns="R2_SD")
    summary.to_csv(path, index=False, float_format="%.17g")
    with pytest.raises(ValueError):
        verify_run(reference_run)


def test_missing_summary_is_rejected(reference_run):
    (reference_run / "summary.csv").unlink()
    with pytest.raises(ValueError, match="missing summary.csv"):
        verify_run(reference_run)


def test_p32_summary_is_checked_without_refitting(tmp_path):
    predictions = pd.read_csv(ROOT / "reference/p32_predictions.csv", float_precision="round_trip")
    keys = ["target", "family", "imputer", "model_seed", "scope"]
    rows = []
    for group, frame in predictions.groupby(keys, sort=False):
        rows.append({**dict(zip(keys, group, strict=True)), **regression_metrics(
            frame.observed_pg_ml, frame.predicted_pg_ml,
        )})
    metrics = pd.DataFrame(rows)
    summary = summarize_metrics(metrics)
    for name, frame in [("predictions", predictions), ("metrics", metrics), ("summary", summary)]:
        frame.to_csv(tmp_path / f"{name}.csv", index=False, float_format="%.17g")
    report = verify_run(tmp_path)
    assert report["summary_rows"] == 2
    assert report["max_summary_difference"] == 0
    summary.iloc[:1].to_csv(tmp_path / "summary.csv", index=False, float_format="%.17g")
    with pytest.raises(ValueError, match="Summary identities"):
        verify_run(tmp_path)
