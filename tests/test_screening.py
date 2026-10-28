import json

import numpy as np
import pandas as pd
import pytest

from hydrogel.data import load_schema
from hydrogel.screening import (
    CandidateLibrary, candidate_form, constraint_options, front, load_library,
    normalise_constraints, result_payload, screen,
)


def synthetic(tnf, il10, **overrides):
    schema = load_schema()
    tnf, il10 = np.asarray(tnf, dtype=float), np.asarray(il10, dtype=float)
    if tnf.ndim == 1:
        tnf = np.repeat(tnf[:, None], 3, axis=1)
    if il10.ndim == 1:
        il10 = np.repeat(il10[:, None], 3, axis=1)
    n = len(tnf)
    columns = {name: np.asarray([next(iter(details["labels"]))] * n, dtype="S1")
               for name, details in schema["categorical"].items()}
    columns.update({name: np.ones(n) for name in schema["numeric"]})
    columns.update({"candidate_id": np.asarray([f"candidate-{i}" for i in range(n)], dtype="S20"),
                    "eligible": np.ones(n, dtype=bool)})
    members = {}
    for target, values in [("tnf", tnf), ("il10", il10)]:
        members[target] = [f"{target}_member_{i}" for i in range(3)]
        columns.update({name: values[:, i] for i, name in enumerate(members[target])})
        columns[target + "_mean_pg_ml"] = values.mean(axis=1)
        columns[target + "_sample_SD_pg_ml"] = values.std(axis=1, ddof=1)
        columns[target + "_AD_ratio"] = np.full(n, .5)
        columns[target + "_all_categories_observed_in_Training"] = np.ones(n, dtype=bool)
    for name, values in overrides.items():
        columns[name] = np.asarray(values, dtype="S1" if name in schema["categorical"] else None)
    manifest = {"source_csv_sha256": "synthetic-library", "features": schema["predictors"],
                "member_prediction_columns": members}
    return CandidateLibrary(columns, manifest, schema)


def brute_force_front(x, y):
    return np.asarray([not any((a <= x[i] and b >= y[i] and (a < x[i] or b > y[i]))
                              for a, b in zip(x, y)) for i in range(len(x))])


def test_sorting_front_matches_exact_dominance_and_ties():
    x = np.asarray([1, 1, 2, 2, 3, 4, 4, 5, 1, np.nextafter(1., 2.)])
    y = np.asarray([2, 2, 3, 2, 2, 4, 4, 3, 1, 2])
    np.testing.assert_array_equal(front(x, y), brute_force_front(x, y))
    for length in range(1, 60):
        generator = np.random.default_rng(length)
        x, y = generator.integers(0, 10, size=(2, length))
        np.testing.assert_array_equal(front(x, y), brute_force_front(x, y))


@pytest.mark.parametrize("x,y", [([1], [np.inf]), ([np.nan], [1]), ([1, 2], [1]), ([[1]], [[1]])])
def test_front_rejects_invalid_objectives(x, y):
    with pytest.raises(ValueError):
        front(x, y)


def test_empty_single_and_exact_tied_profiles():
    library = synthetic([1, 1, 2], [4, 4, 3])
    result = screen(library, {})
    assert result.retained_indices.tolist() == [0, 1]
    assert result.retained_frequencies.tolist() == [9, 9]
    empty = screen(library, {"Gprime_kPa": [100, None]})
    assert len(empty.matching_indices) == 0
    assert len(empty.retained_indices) == 0
    assert result_payload(library, empty)["retained"] == []
    singleton = screen(synthetic([1], [2]), {})
    assert singleton.retained_indices.tolist() == [0]


def test_globally_dominated_candidate_becomes_conditional_front():
    library = synthetic([1, 2], [2, 1], macrophage_model_code=["A", "B"])
    assert screen(library, {}).retained_indices.tolist() == [0]
    conditional = screen(library, {"macrophage_model_code": "B"})
    assert conditional.matching_indices.tolist() == [1]
    assert conditional.retained_indices.tolist() == [1]
    assert conditional.retained_frequencies.tolist() == [9]


def test_exact_category_time_and_full_precision_range_boundaries():
    value = 1.2345678901234567
    library = synthetic([1, 2, 3], [1, 2, 3],
                        Gprime_kPa=[value, np.nextafter(value, np.inf), value],
                        pore_size_um=[3., 3., 4.], exposure_time_h=[24., 24., 48.],
                        bioactive_cue_code=["A", "A", "B"])
    result = screen(library, {"Gprime_kPa": [str(value), str(value)], "pore_size_um": [3., 3.],
                              "exposure_time_h": 24., "bioactive_cue_code": "A"})
    assert result.matching_indices.tolist() == [0]
    assert len(screen(library, {"exposure_time_h": 24.000000000000004}).matching_indices) == 0
    assert len(screen(library, {"bioactive_cue_code": "F"}).matching_indices) == 0


@pytest.mark.parametrize("constraints", [
    {"Gprime_kPa": [-1, 2]}, {"pore_size_um": [1, float("inf")]},
    {"Gprime_kPa": [True, 2]}, {"Gprime_kPa": ["bad", 2]},
    {"Gprime_kPa": [2, 1]}, {"Gprime_kPa": 1},
    {"exposure_time_h": True}, {"exposure_time_h": "not a number"},
    {"exposure_time_h": 10**400},
    {"macrophage_model_code": "unknown"}, {"macrophage_model_code": True},
    {"new_descriptor": 5},
])
def test_constraints_reject_invalid_values(constraints):
    with pytest.raises(ValueError):
        normalise_constraints(synthetic([1], [1]), constraints)


def test_unbounded_ranges_do_not_relax_the_supplied_bound():
    library = synthetic([1, 2], [1, 2], Gprime_kPa=[1., 2.])
    result = screen(library, {"Gprime_kPa": ["1.5", None]})
    assert result.constraints["Gprime_kPa"] == [1.5, None]
    assert result.matching_indices.tolist() == [1]
    assert normalise_constraints(library, {"Gprime_kPa": ["", None]})["Gprime_kPa"] is None


def test_pairing_consistency_uses_member_predictions_in_the_same_subset():
    library = synthetic([[0, 10, 10], [7, 7, 7]], [2, 3], macrophage_model_code=["A", "B"])
    result = screen(library, {})
    assert result.mean_front_indices.tolist() == [0, 1]
    assert result.retained_indices.tolist() == [1]
    subset = screen(library, {"macrophage_model_code": "A"})
    assert subset.retained_indices.tolist() == [0]
    assert subset.retained_frequencies.tolist() == [9]
    boundary = screen(synthetic([[0, 0, 20], [7, 7, 7]], [2, 3]), {})
    assert boundary.retained_frequencies.tolist() == [6, 9]


def test_no_candidate_passes_consistency_without_lowering_threshold():
    library = synthetic([[0, 10, 10], [10, 0, 10], [10, 10, 0]], [2, 2, 2])
    result = screen(library, {})
    assert len(result.mean_front_indices) == 3
    assert len(result.retained_indices) == 0
    assert len(result.matching_indices) == 3


def test_download_and_form_transfer_preserve_exact_values():
    exact = 1.2345678901234567
    library = synthetic([1, 2], [1, 2], Gprime_kPa=[exact, 2.])
    result = screen(library, {"exposure_time_h": 1.})
    payload = result_payload(library, result, max_plot_points=1)
    assert payload["plot"]["sampled"]
    assert len(payload["retained"]) == 2
    assert len(payload["plot"]["candidates"]) == 1
    values = candidate_form(library, "candidate-0")
    assert values["Gprime_kPa"] == exact
    assert list(values) == library.schema["predictors"]
    assert payload["retained"][0]["values"] == values
    from io import StringIO
    table = pd.read_csv(StringIO(payload["downloads"]["profiles_csv"]), float_precision="round_trip")
    assert table.Gprime_kPa.iloc[0] == exact
    assert len(table) == 2
    assert json.loads(payload["downloads"]["constraints_json"])["constraints"] == result.constraints


@pytest.fixture(scope="module")
def recorded_library():
    return load_library()


def test_complete_library_reproduces_canonical_global_profiles(recorded_library):
    library = recorded_library
    result = screen(library, {})
    ids = library.columns["candidate_id"][result.retained_indices].astype(str).tolist()
    assert ids == library.manifest["canonical_retained_candidate_ids"]
    assert len(library) == 722000
    assert int(library.columns["prediction_evaluated"].sum()) == 695722
    assert len(result.matching_indices) == 656792
    assert len(result.mean_front_indices) == 24
    assert len(result.retained_indices) == 13
    assert all(not column.flags.writeable for column in library.columns.values())


def test_library_options_use_observed_codes_and_exact_grid(recorded_library):
    library = recorded_library
    options = constraint_options(library)
    for field, values in options["categorical"].items():
        assert {item["value"] for item in values} == set(library.columns[field].astype(str))
        assert all(item["label"] == library.schema["categorical"][field]["labels"][item["value"]] for item in values)
    for field, values in options["numeric"].items():
        np.testing.assert_array_equal(values["values"], np.unique(library.columns[field]))


def test_stored_candidate_predictions_match_saved_model_inference(recorded_library):
    from hydrogel.application import load_resources
    from threadpoolctl import threadpool_limits

    library = recorded_library
    resources = load_resources()
    result = screen(library, {"macrophage_model_code": "A", "activation_stimulus_code": "A",
                              "exposure_mode_code": "E", "exposure_time_h": 24.})
    ids = list(dict.fromkeys(library.manifest["canonical_retained_candidate_ids"][::6]
                           + library.columns["candidate_id"][result.retained_indices[:3]].astype(str).tolist()))
    indices = np.asarray([np.flatnonzero(library.columns["candidate_id"] == value.encode())[0] for value in ids])
    frame = library.frame(indices, library.schema["predictors"])
    with threadpool_limits(limits=1):
        for target, resource in resources["targets"].items():
            for saved in resource["models"]:
                expected = library.columns[f"{target}_seed_{saved['seed']}_pg_ml"][indices]
                np.testing.assert_allclose(saved["pipeline"].predict(frame), expected, rtol=0, atol=1e-7)
