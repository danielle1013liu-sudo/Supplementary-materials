"""CSV and form measurements obey the same parsing and missingness rules."""

import numpy as np
import pandas as pd
import pytest

from hydrogel.application import form_frame
from hydrogel.data import load_c4, load_schema, validate_features


@pytest.fixture
def form():
    return load_c4("tnf").iloc[0][load_schema()["predictors"]].to_dict()


@pytest.mark.parametrize("name", ["Gprime_kPa", "pore_size_um", "exposure_time_h"])
@pytest.mark.parametrize("value", [
    -0.01, "-1", np.inf, -np.inf, "inf", "NaN", True, False, np.bool_(True),
    "true", "invalid", "1 kPa", [], {}, 1 + 2j, 10 ** 400,
])
def test_csv_and_form_reject_invalid_measurements(form, name, value):
    form[name] = value
    with pytest.raises(ValueError, match=name):
        form_frame(form)
    # object dtype prevents pandas from silently converting booleans before validation.
    frame = pd.DataFrame([form], dtype=object)
    with pytest.raises(ValueError, match=name):
        validate_features(frame, load_schema())


@pytest.mark.parametrize("name", ["Gprime_kPa", "pore_size_um", "exposure_time_h"])
@pytest.mark.parametrize("value", [None, np.nan, np.float32("nan"), pd.NA, "", "  ", "Unreported", "—"])
def test_csv_and_form_preserve_explicit_missing_values(form, name, value):
    form[name] = value
    actual_form = form_frame(form)
    actual_csv = validate_features(pd.DataFrame([form], dtype=object), load_schema())
    assert pd.isna(actual_form.loc[0, name])
    assert pd.isna(actual_csv.loc[0, name])
    assert actual_form.loc[0, "hydrogel_composition_code"] == form["hydrogel_composition_code"]


@pytest.mark.parametrize("value", [0, 0.0, "0", 0.12345678901234567, " 12.375 ", np.int64(24)])
def test_valid_measurements_are_not_rounded_or_imputed(form, value):
    form["Gprime_kPa"] = value
    actual_form = form_frame(form)
    actual_csv = validate_features(pd.DataFrame([form], dtype=object), load_schema())
    assert actual_form.loc[0, "Gprime_kPa"] == float(value)
    assert actual_csv.loc[0, "Gprime_kPa"] == float(value)


def test_validation_does_not_modify_input_frame(form):
    form["Gprime_kPa"] = "Unreported"
    original = pd.DataFrame([form])
    unchanged = original.copy(deep=True)
    validate_features(original, load_schema())
    pd.testing.assert_frame_equal(original, unchanged)
