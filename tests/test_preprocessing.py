"""Preprocessing contracts that affect reproducibility and feature meaning."""

import numpy as np
import pandas as pd
import pytest

from hydrogel.data import load_schema, validate_features
from hydrogel.models import make_estimator
from hydrogel.preprocessing import ScaledKNNImputer, make_preprocessor


@pytest.fixture
def schema():
    return load_schema()


@pytest.fixture
def training_frame(schema):
    values = {}
    for name in schema["predictors"]:
        if name in schema["numeric"]:
            values[name] = [2.0, 4.0, 6.0]
        else:
            first = next(iter(schema["categorical"][name]["labels"]))
            values[name] = [first] * 3
    return pd.DataFrame(values)


def test_imputation_uses_training_statistics_and_preserves_units(
    schema, training_frame
):
    preprocessor = make_preprocessor(schema, "median").fit(training_frame)
    query = training_frame.iloc[[0]].copy()
    query.loc[:, schema["numeric"][0]] = np.nan
    query.loc[:, schema["numeric"][1]] = 1_000_000.0
    transformed = preprocessor.transform(query)
    np.testing.assert_array_equal(transformed[0, :3], [4.0, 1_000_000.0, 2.0])


def test_one_hot_order_includes_schema_levels_absent_from_training(
    schema, training_frame
):
    preprocessor = make_preprocessor(schema, "median").fit(training_frame)
    names = schema["numeric"] + [
        f"{name}_{code}"
        for name in schema["predictors"]
        if name in schema["categorical"]
        for code in schema["categorical"][name]["labels"]
    ]
    assert len(names) == 35
    assert preprocessor.get_feature_names_out().tolist() == names
    query = training_frame.iloc[[0]].copy()
    query.loc[:, "hydrogel_composition_code"] = "C"
    transformed = preprocessor.transform(query)
    assert transformed.shape == (1, 35)
    assert transformed[0, names.index("hydrogel_composition_code_C")] == 1
    assert transformed[0, names.index("hydrogel_composition_code_A")] == 0


def test_unknown_category_is_rejected(schema, training_frame):
    preprocessor = make_preprocessor(schema, "median").fit(training_frame)
    query = training_frame.iloc[[0]].copy()
    query.loc[:, "bioactive_cue_code"] = "unknown"
    with pytest.raises(ValueError, match="unknown categories"):
        preprocessor.transform(query)
    with pytest.raises(ValueError, match="Unknown or missing categories"):
        validate_features(query, schema)


def test_knn_restores_observed_values_exactly():
    training = np.array([[0.125, 1000.0], [1.3, 2000.0], [2.8, 3100.0]])
    query = np.array([[np.nan, 1100.123456789], [1.234567890123, np.nan]])
    result = ScaledKNNImputer(n_neighbors=3).fit(training).transform(query)
    observed = ~np.isnan(query)
    np.testing.assert_array_equal(result[observed], query[observed])
    assert np.isfinite(result).all()
    assert 0.125 <= result[0, 0] <= 2.8
    assert 1000.0 <= result[1, 1] <= 3100.0


def test_knn_rejects_an_unidentifiable_training_column():
    with pytest.raises(ValueError, match="entirely missing"):
        ScaledKNNImputer().fit([[1.0, np.nan], [2.0, np.nan]])


@pytest.mark.parametrize("family", ["GBR", "RF", "XGB", "LGBM"])
def test_replicate_seed_is_applied_to_each_estimator(family):
    parameters = {"n_estimators": 3}
    model = make_estimator(family, parameters, 1234)
    assert parameters == {"n_estimators": 3}
    assert model.get_params()["random_state"] == 1234
    if family != "GBR":
        assert model.get_params()["n_jobs"] == 1
    if family == "LGBM":
        for name in (
            "bagging_seed", "feature_fraction_seed", "data_random_seed", "drop_seed"
        ):
            assert model.get_params()[name] == 1234
        assert model.get_params()["deterministic"] is True


def test_xgboost_null_missing_parameter_becomes_nan():
    model = make_estimator("XGB", {"missing": None}, 7)
    assert np.isnan(model.get_params()["missing"])
