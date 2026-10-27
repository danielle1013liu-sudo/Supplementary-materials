"""Check deployment entrypoints and the form-to-model interface."""

import pytest
import streamlit as st
import streamlit.components.v1 as components
from streamlit.testing.v1 import AppTest

from hydrogel import application
from hydrogel.data import ROOT, load_c4, load_schema


@pytest.fixture
def component_capture(monkeypatch):
    state = {"event": None, "arguments": None}

    def component(**kwargs):
        state["arguments"] = kwargs
        return state["event"]

    monkeypatch.setattr(components, "declare_component", lambda *args, **kwargs: component)
    st.cache_resource.clear()
    return state


@pytest.mark.parametrize("entrypoint", ["app.py", "deploy/streamlit_app.py"])
def test_entrypoints_use_current_descriptor_controls(component_capture, entrypoint):
    app = AppTest.from_file(str(ROOT / entrypoint), default_timeout=30).run()
    assert not app.exception
    controls = component_capture["arguments"]
    schema = load_schema()
    assert [field["machine_column"] for field in controls["fields"]] == list(schema["categorical"])
    for field in controls["fields"]:
        assert [item["machine_value"] for item in field["categories"]] == list(
            schema["categorical"][field["machine_column"]]["labels"]
        )
    assert list(controls["numeric"]) == schema["numeric"]
    assert controls["numeric"]["Gprime_kPa"]["display_name"] == "Storage modulus G′"
    assert controls["codebook"].startswith("data:application/json;base64,")


def _prediction_event():
    values = load_c4("tnf").iloc[0][load_schema()["predictors"]].to_dict()
    return {"id": "test-prediction", "action": "predict", "form": {"values": values, "unreported": {}}}


def test_unreported_numeric_value_is_missing_without_overwriting_the_form(component_capture, monkeypatch):
    event = _prediction_event()
    event["form"]["unreported"]["Gprime_kPa"] = True
    event["form"]["values"]["Gprime_kPa"] = "987"
    component_capture["event"] = event
    received = []
    monkeypatch.setattr(application, "load_resources", lambda root: {})

    def predict(values, resources):
        received.append(values)
        return {"targets": [], "warnings": [], "downloads": {}}

    monkeypatch.setattr(application, "predict_form", predict)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert len(received) == 1
    assert received[0]["Gprime_kPa"] is None
    assert app.session_state["form"]["values"]["Gprime_kPa"] == "987"


def test_blank_numeric_value_requires_explicit_unreported_selection(component_capture, monkeypatch):
    event = _prediction_event()
    event["form"]["values"]["Gprime_kPa"] = ""
    component_capture["event"] = event

    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid form values must not reach model inference.")

    monkeypatch.setattr(application, "load_resources", unexpected)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert "select Unreported" in app.session_state["error"]
