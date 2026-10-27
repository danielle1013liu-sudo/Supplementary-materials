from copy import deepcopy
from io import StringIO

import numpy as np
import pandas as pd
import pytest

from hydrogel.application import _domain, form_frame, load_resources, predict_form
from hydrogel.data import ROOT, load_c4


@pytest.fixture(scope="module")
def resources():
    return load_resources(ROOT)


@pytest.fixture
def form(resources):
    return load_c4("tnf", ROOT).iloc[0][resources["schema"]["predictors"]].to_dict()


def test_deployment_uses_main_training_models_and_domains(resources):
    for target, expected_n in [("tnf", 120), ("il10", 104)]:
        saved = resources["targets"][target]
        assert saved["training_n"] == expected_n
        assert saved["diagnostic"]["reference"].shape == (expected_n, 35)
        assert len(saved["models"]) == 3
    assert resources["targets"]["tnf"]["family"] == "RF"
    assert resources["targets"]["il10"]["family"] == "LGBM"
    assert resources["targets"]["tnf"]["diagnostic"]["threshold"] == pytest.approx(3.116692400813483)
    assert resources["targets"]["il10"]["diagnostic"]["threshold"] == pytest.approx(2.9470710137702048)


@pytest.mark.parametrize("row", [0, 1])
def test_c4_predictions_match_references_with_additive_contributions(resources, row):
    form = load_c4("tnf", ROOT).iloc[row][resources["schema"]["predictors"]].to_dict()
    identity = load_c4("tnf", ROOT).iloc[row].record_id
    output = predict_form(form, resources)
    reference = pd.read_csv(ROOT / "reference/predictions.csv.gz", float_precision="round_trip")
    for result in output["targets"]:
        saved = resources["targets"][result["target"]]
        expected = reference.loc[
            (reference.target == result["target"]) & (reference.family == saved["family"])
            & (reference.imputer == saved["imputer"]) & (reference.scope == "C4")
            & (reference.record_id == identity)
        ].set_index("model_seed").loc[saved["model_seeds"], "predicted_pg_ml"].to_numpy()
        actual = np.array([individual["value"] for individual in result["individual"]])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
        assert result["mean"] == pytest.approx(expected.mean(), abs=1e-7)
        assert result["sd"] == pytest.approx(expected.std(ddof=1), abs=1e-7)
        assert len(result["contributions"]) == 10
        reconstructed = result["expected_value_pg_ml"] + sum(item["SHAP_pg_ml"] for item in result["contributions"])
        assert reconstructed == pytest.approx(result["mean"], abs=1e-7)
        assert result["domain_ratio"] == result["mean5_distance"] / result["threshold_q95"]
        assert result["domain_pass"] == (result["numerical_AD_pass"] and result["categorical_support_pass"])
        assert len(result["nearest_neighbours"]) == 5
    downloads = output["downloads"]
    assert len(pd.read_csv(StringIO(downloads["inputs_csv"]))) == 1
    assert len(pd.read_csv(StringIO(downloads["predictions_csv"]))) == 6
    assert len(pd.read_csv(StringIO(downloads["contributions_csv"]))) == 20


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True, "", "invalid"])
def test_invalid_numerical_input_is_rejected(form, value):
    form["Gprime_kPa"] = value
    with pytest.raises(ValueError, match="Gprime_kPa"):
        form_frame(form, ROOT)


def test_unreported_input_remains_missing(form, resources):
    form["Gprime_kPa"] = None
    frame = form_frame(form, ROOT)
    assert pd.isna(frame.loc[0, "Gprime_kPa"])
    output = predict_form(form, resources)
    assert any("imputed" in warning for warning in output["warnings"])
    assert pd.isna(pd.read_csv(StringIO(output["downloads"]["inputs_csv"])).loc[0, "Gprime_kPa"])
    assert all(np.isfinite(item["mean"]) for item in output["targets"])


def test_unknown_category_is_rejected(form):
    form["bioactive_cue_code"] = "Z"
    with pytest.raises(ValueError, match="Unknown category"):
        form_frame(form, ROOT)


def test_categorical_support_requires_observation_in_training(form, resources):
    frame = form_frame(form, ROOT)
    diagnostic = deepcopy(resources["targets"]["tnf"]["diagnostic"])
    descriptor = "bioactive_cue_code"
    diagnostic["supported_levels"][descriptor].discard(form[descriptor])
    result = _domain(frame, diagnostic, resources["schema"])
    assert result["numerical_AD_pass"]
    assert not result["categorical_support_pass"]
    assert not result["domain_pass"]
    assert result["unsupported_levels"][0]["descriptor"] == descriptor


def test_domain_distance_can_reject_supported_categories(form, resources):
    form["Gprime_kPa"] = 1000000
    frame = form_frame(form, ROOT)
    result = _domain(frame, resources["targets"]["tnf"]["diagnostic"], resources["schema"])
    assert result["categorical_support_pass"]
    assert result["domain_ratio"] > 1
    assert not result["domain_pass"]


def test_loading_resources_never_fits_estimators(monkeypatch):
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer, SimpleImputer
    from sklearn.preprocessing import StandardScaler

    def unexpected_fit(*args, **kwargs):
        pytest.fail("Application loading must not fit estimators or preprocessors.")

    for estimator in [LGBMRegressor, RandomForestRegressor, IterativeImputer, SimpleImputer, StandardScaler]:
        monkeypatch.setattr(estimator, "fit", unexpected_fit)
    loaded = load_resources(ROOT)
    assert set(loaded["targets"]) == {"tnf", "il10"}


def test_runtime_prediction_drift_prevents_loading(monkeypatch):
    from sklearn.ensemble import RandomForestRegressor

    predict = RandomForestRegressor.predict

    def changed_prediction(self, features):
        return predict(self, features) + 1.0

    monkeypatch.setattr(RandomForestRegressor, "predict", changed_prediction)
    with pytest.raises(ValueError, match="recorded predictions on this runtime"):
        load_resources(ROOT)
