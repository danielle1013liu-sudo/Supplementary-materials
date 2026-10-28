"""Screening events preserve server-owned candidate values and current state."""

import json
from copy import deepcopy

import numpy as np
import pytest
import streamlit as st
import streamlit.components.v1 as components
from streamlit.testing.v1 import AppTest

from hydrogel import application, screening
from hydrogel.data import ROOT, load_schema


@pytest.fixture
def screening_app(monkeypatch):
    schema = load_schema()
    columns = {
        name: np.array([next(iter(details["labels"]))] * 3, dtype="S1")
        for name, details in schema["categorical"].items()
    }
    columns.update({
        "candidate_id": np.array(["profile-precise", "profile-restricted", "profile-second"], dtype="S20"),
        "macrophage_model_code": np.array(["A", "B", "A"], dtype="S1"),
        "activation_stimulus_code": np.array(["A", "A", "B"], dtype="S1"),
        "Gprime_kPa": np.array([1.2345678901234567, 2., 4.]),
        "pore_size_um": np.array([3.3333333333333335, 5., 7.]),
        "exposure_time_h": np.array([24., 24., 48.]),
        "eligible": np.ones(3, dtype=bool),
    })
    members = {}
    for target, values in [("tnf", [2., 3., 4.]), ("il10", [4., 3., 6.])]:
        members[target] = [f"{target}_member_{index}" for index in range(3)]
        for name in members[target]:
            columns[name] = np.asarray(values)
        columns[target + "_mean_pg_ml"] = np.asarray(values)
        columns[target + "_sample_SD_pg_ml"] = np.zeros(3)
        columns[target + "_AD_ratio"] = np.full(3, 0.5)
        columns[target + "_all_categories_observed_in_Training"] = np.ones(3, dtype=bool)
    manifest = {"source_csv_sha256": "application-test-library", "features": schema["predictors"],
                "member_prediction_columns": members}
    library = screening.CandidateLibrary(columns, manifest, schema)
    captured = {"event": None, "arguments": None, "screen_calls": 0, "predictions": []}

    def component(**kwargs):
        captured["arguments"] = kwargs
        return captured["event"]

    original_screen = screening.screen

    def counted_screen(*args, **kwargs):
        captured["screen_calls"] += 1
        return original_screen(*args, **kwargs)

    def predict(values, resources):
        captured["predictions"].append(deepcopy(values))
        return {"targets": [], "warnings": [], "downloads": {}}

    monkeypatch.setattr(components, "declare_component", lambda *args, **kwargs: component)
    monkeypatch.setattr(screening, "load_library", lambda root: library)
    monkeypatch.setattr(screening, "library_manifest", lambda root: manifest)
    monkeypatch.setattr(screening, "screen", counted_screen)
    monkeypatch.setattr(application, "load_resources", lambda root: {})
    monkeypatch.setattr(application, "predict_form", predict)
    st.cache_resource.clear()
    st.cache_data.clear()
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    yield app, captured, library
    st.cache_resource.clear()
    st.cache_data.clear()


def submit(app, captured, action, identity, **values):
    captured["event"] = {"id": identity, "action": action, **values}
    app.run()
    assert not app.exception


def test_screening_and_prediction_keep_separate_results(screening_app):
    app, captured, library = screening_app
    form = {"values": screening.candidate_form(library, "profile-precise"), "unreported": {}}
    submit(app, captured, "predict", "predict-1", form=form)
    prediction = app.session_state["result"]
    submit(app, captured, "screen", "screen-1", constraints={"macrophage_model_code": "B"})
    result = app.session_state["screening_result"]
    assert result["matching_count"] == 1
    assert [row["candidate_id"] for row in result["retained"]] == ["profile-restricted"]
    assert app.session_state["result"] == prediction
    constraints = json.loads(result["downloads"]["constraints_json"])["constraints"]
    assert constraints["macrophage_model_code"] == "B"
    assert captured["arguments"]["screening_response_id"] == "screen-1"


def test_loading_profile_uses_exact_server_values_and_existing_inference(screening_app):
    app, captured, library = screening_app
    submit(app, captured, "screen", "screen-1", constraints={})
    expected = screening.candidate_form(library, "profile-precise")
    submit(app, captured, "load_candidate", "load-1", candidate_id="profile-precise",
           constraints={}, form={"values": {"Gprime_kPa": 9999}})
    form = app.session_state["form"]
    assert form == {"values": expected, "unreported": {}}
    assert form["values"]["Gprime_kPa"] == 1.2345678901234567
    assert app.session_state["result"] is None
    assert app.session_state["transfer_response_id"] == "load-1"
    submit(app, captured, "predict", "predict-1", form=form)
    assert captured["predictions"] == [expected]


@pytest.mark.parametrize("action", ["screening_change", "screening_reset"])
def test_changed_conditions_clear_results_and_reject_stale_selection(screening_app, action):
    app, captured, _ = screening_app
    submit(app, captured, "screen", "screen-1", constraints={})
    assert app.session_state["screening_result"]["retained_count"] == 2
    submit(app, captured, action, "change-1", constraints={"macrophage_model_code": "B"})
    assert app.session_state["screening_result"] is None
    assert app.session_state["screening_error"] is None
    submit(app, captured, "load_candidate", "load-1", candidate_id="profile-precise", constraints={})
    assert "current screening results" in app.session_state["screening_error"]
    assert app.session_state["form"] is None


def test_transfer_rejects_changed_conditions_even_before_change_event(screening_app):
    app, captured, _ = screening_app
    submit(app, captured, "screen", "screen-1", constraints={})
    submit(app, captured, "load_candidate", "load-1", candidate_id="profile-precise",
           constraints={"macrophage_model_code": "B"})
    assert "conditions have changed" in app.session_state["screening_error"]
    assert app.session_state["form"] is None


def test_transfer_rejects_candidates_outside_the_current_retained_set(screening_app):
    app, captured, _ = screening_app
    submit(app, captured, "screen", "screen-1", constraints={})
    submit(app, captured, "load_candidate", "load-1", candidate_id="profile-restricted", constraints={})
    assert "current screening results" in app.session_state["screening_error"]


def test_empty_and_invalid_requests_replace_previous_results(screening_app):
    app, captured, _ = screening_app
    submit(app, captured, "screen", "screen-1", constraints={})
    submit(app, captured, "screen", "screen-2", constraints={"Gprime_kPa": [1000, None]})
    assert app.session_state["screening_result"]["matching_count"] == 0
    assert app.session_state["screening_result"]["retained_count"] == 0
    submit(app, captured, "screen", "screen-3", constraints={"pore_size_um": [-1, 2]})
    assert app.session_state["screening_result"] is None
    assert "pore_size_um" in app.session_state["screening_error"]


def test_equivalent_requests_reuse_cache_and_constraints_do_not(screening_app):
    app, captured, _ = screening_app
    submit(app, captured, "screen", "screen-1", constraints={"exposure_time_h": 24})
    submit(app, captured, "screen", "screen-2", constraints={"exposure_time_h": "24"})
    assert captured["screen_calls"] == 1
    submit(app, captured, "screen", "screen-3", constraints={"exposure_time_h": 48})
    assert captured["screen_calls"] == 2
    assert app.session_state["screening_result"]["constraints"]["exposure_time_h"] == 48


def test_existing_session_initialises_new_constraint_state(screening_app):
    app, captured, _ = screening_app
    release = app.session_state["release_key"]
    del app.session_state["screening_constraints"]
    app.run()
    assert not app.exception
    assert app.session_state["release_key"] == release
    assert app.session_state["screening_constraints"] is None
    assert captured["arguments"]["screening_constraints"] is None
