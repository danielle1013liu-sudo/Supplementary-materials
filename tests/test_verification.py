import pandas as pd
import pytest

from hydrogel.data import ROOT
from hydrogel.verification import verify_run


@pytest.fixture
def reference_run(tmp_path):
    for filename in ["predictions.csv.gz", "metrics.csv"]:
        frame = pd.read_csv(ROOT / "reference" / filename, float_precision="round_trip")
        frame = frame.loc[(frame.target == "tnf") & (frame.family == "RF") & (frame.imputer == "median")]
        frame.to_csv(tmp_path / filename.removesuffix(".gz"), index=False, float_format="%.17g")
    return tmp_path


def test_reference_comparison_checks_all_rows(reference_run):
    report = verify_run(reference_run)
    assert report["verified"]
    assert report["prediction_rows"] == 3 * (120 + 120 + 30 + 2)
    assert report["max_prediction_difference"] == 0


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
