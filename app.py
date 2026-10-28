"""Interactive predictions from the released hydrogel cytokine models."""

import base64
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hydrogel.application import load_resources, predict_form
from hydrogel.codebook import codebook_json
from hydrogel.data import load_dataset, load_schema
from hydrogel.screening import (
    candidate_form, constraint_options, library_manifest, load_library,
    normalise_constraints, result_payload, screen,
)


APP_NAME = os.getenv("HYDROGEL_APP_NAME", "Hydrogel Cytokine Explorer")
CATEGORY_LABELS = {
    "hydrogel_composition_code": "Hydrogel composition",
    "bioactive_cue_code": "Bioactive cue",
    "crosslink_method_code": "Crosslinking method",
    "crosslink_level_code": "Crosslinking level",
    "macrophage_model_code": "Macrophage model",
    "activation_stimulus_code": "Activation stimulus",
    "exposure_mode_code": "Exposure mode",
}
NUMERIC_LABELS = {
    "Gprime_kPa": ("Storage modulus G′", "kPa"),
    "pore_size_um": ("Pore size", "µm"),
    "exposure_time_h": ("Exposure time", "h"),
}

st.set_page_config(page_title=APP_NAME, layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """<style>
    [data-testid="stHeader"], footer {display: none}
    .stMainBlockContainer {padding: 0 !important; max-width: none !important}
    .stApp {background: #F6F0F6}
    [data-testid="stElementContainer"] {margin: 0}
    [data-testid="stVerticalBlock"] {gap: 0 !important}
    iframe {display: block; border: 0}
    </style>""",
    unsafe_allow_html=True,
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_uri(path: Path, mime: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def text_uri(value: str, mime: str) -> str:
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@st.cache_resource
def resources(release_key: tuple[str, ...]) -> dict:
    """Invalidate loaded models whenever the release manifests change."""
    return load_resources(ROOT)


@st.cache_resource(max_entries=1, show_spinner=False)
def candidate_library(version: str):
    """Share the verified, read-only library without sending it to the browser."""
    return load_library(ROOT)


@st.cache_data(max_entries=1, show_spinner=False)
def screening_controls(version: str) -> dict:
    return constraint_options(candidate_library(version))


@st.cache_data(max_entries=16, show_spinner=False)
def screening_results(version: str, constraints_json: str) -> dict:
    """Cache compact results by the complete asset manifest and exact restrictions."""
    library = candidate_library(version)
    selected = screen(library, json.loads(constraints_json))
    return result_payload(library, selected)


@st.cache_data
def descriptor_controls(release_key: tuple[str, ...]) -> tuple[list, dict]:
    schema = load_schema(ROOT)
    fields = [
        {
            "machine_column": name,
            "feature": CATEGORY_LABELS[name],
            "categories": [
                {"machine_value": code, "new_code": code, "display_label": label}
                for code, label in details["labels"].items()
            ],
        }
        for name, details in schema["categorical"].items()
    ]
    observations = pd.concat([load_dataset(target, ROOT) for target in ("tnf", "il10")])
    numeric = {}
    for name in schema["numeric"]:
        label, units = NUMERIC_LABELS[name]
        numeric[name] = {
            "display_name": label,
            "units": units,
            "minimum": float(observations[name].min()),
            "maximum": float(observations[name].max()),
        }
    return fields, numeric


release_key = tuple(file_hash(ROOT / name) for name in (
    "data/manifest.json", "deployment_assets/manifest.json",
))
screening_version = file_hash(ROOT / "deployment_assets/screening/manifest.json")
interface_version = (*release_key, screening_version)
if st.session_state.get("release_key") != interface_version:
    st.session_state.release_key = interface_version
    for name in (
        "form", "result", "error", "last_event", "screening_result",
        "screening_error", "screening_constraints", "prediction_response_id", "screening_response_id",
        "transfer_response_id",
    ):
        st.session_state[name] = None

# Initialise new interface state for sessions that survive a code update.
st.session_state.setdefault("screening_constraints", None)

fields, numeric = descriptor_controls(release_key)
frontend_version = hashlib.sha256(
    b"".join((ROOT / "web" / asset).read_bytes() for asset in (
        "index.html", "app.js", "screening-state.js", "style.css",
    ))
).hexdigest()[:12]
component = components.declare_component(
    f"hydrogel_explorer_{frontend_version}", path=str(ROOT / "web"),
)
mobile_banner = ROOT / "assets/banner_mobile.svg"
event = component(
    name=APP_NAME,
    fields=fields,
    numeric=numeric,
    form=st.session_state.form,
    result=st.session_state.result,
    error=st.session_state.error,
    banner=file_uri(ROOT / "assets/banner.svg", "image/svg+xml"),
    banner_mobile=file_uri(mobile_banner, "image/svg+xml") if mobile_banner.is_file() else None,
    codebook=text_uri(codebook_json(ROOT), "application/json"),
    codebook_filename="hydrogel_descriptor_codebook.json",
    screening_options=screening_controls(screening_version),
    library_version=library_manifest(ROOT)["source_csv_sha256"],
    screening_result=st.session_state.screening_result,
    screening_constraints=st.session_state.screening_constraints,
    screening_error=st.session_state.screening_error,
    prediction_response_id=st.session_state.prediction_response_id,
    screening_response_id=st.session_state.screening_response_id,
    transfer_response_id=st.session_state.transfer_response_id,
    key="hydrogel-" + "-".join(value[:12] for value in interface_version),
    default=None,
)
if isinstance(event, dict) and event.get("id") != st.session_state.last_event:
    st.session_state.last_event = event.get("id")
    action = event.get("action")
    if action in {"predict", "change", "reset"}:
        st.session_state.form = event.get("form")
        st.session_state.result = None
        st.session_state.error = None
        st.session_state.prediction_response_id = event.get("id")
    if action == "predict":
        try:
            submitted = event["form"]
            values = dict(submitted["values"])
            for name, (label, _) in NUMERIC_LABELS.items():
                if submitted.get("unreported", {}).get(name, False):
                    values[name] = None
                elif values.get(name) in ("", None):
                    raise ValueError(f"Enter {label} or select Unreported.")
            st.session_state.result = predict_form(values, resources(release_key))
        except (ValueError, AssertionError, OSError, RuntimeError) as error:
            st.session_state.error = str(error) or "The model release could not be verified."
    elif action in {"screen", "screening_change", "screening_reset"}:
        st.session_state.screening_result = None
        st.session_state.screening_error = None
        st.session_state.screening_response_id = event.get("id")
        st.session_state.screening_constraints = (
            {} if action == "screening_reset" else event.get("constraints", {})
        )
        if action == "screen":
            try:
                library = candidate_library(screening_version)
                constraints = normalise_constraints(library, event.get("constraints", {}))
                key = json.dumps(constraints, sort_keys=True, allow_nan=False)
                st.session_state.screening_result = screening_results(screening_version, key)
            except (ValueError, AssertionError, OSError, RuntimeError) as error:
                st.session_state.screening_error = str(error) or "The candidate library could not be verified."
    elif action == "load_candidate":
        st.session_state.screening_error = None
        st.session_state.transfer_response_id = event.get("id")
        try:
            selected = st.session_state.screening_result
            identity = event.get("candidate_id")
            if selected is None or identity not in {row["candidate_id"] for row in selected["retained"]}:
                raise ValueError("Select a retained profile from the current screening results.")
            library = candidate_library(screening_version)
            if "constraints" in event:
                constraints = normalise_constraints(library, event["constraints"])
                if constraints != selected["constraints"]:
                    raise ValueError("The conditions have changed. Run screening before loading a profile.")
            values = candidate_form(library, identity)
            st.session_state.form = {"values": values, "unreported": {}}
            st.session_state.result = None
            st.session_state.error = None
            st.session_state.prediction_response_id = None
        except (ValueError, AssertionError, OSError, RuntimeError) as error:
            st.session_state.screening_error = str(error)
    st.rerun()
