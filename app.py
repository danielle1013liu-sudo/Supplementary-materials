"""Interactive predictions from the released hydrogel cytokine models."""

import base64
import hashlib
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
from hydrogel.data import load_dataset, load_schema


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


@st.cache_resource
def resources(release_key: tuple[str, ...]) -> dict:
    """Invalidate loaded models whenever the release manifests change."""
    return load_resources(ROOT)


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
if st.session_state.get("release_key") != release_key:
    st.session_state.release_key = release_key
    st.session_state.form = None
    st.session_state.result = None
    st.session_state.error = None
    st.session_state.last_event = None

fields, numeric = descriptor_controls(release_key)
frontend_version = hashlib.sha256(
    b"".join((ROOT / "web" / asset).read_bytes() for asset in ("index.html", "app.js", "style.css"))
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
    codebook=file_uri(ROOT / "data/feature_schema.json", "application/json"),
    codebook_filename="hydrogel_descriptor_codebook.json",
    key="hydrogel-" + "-".join(value[:12] for value in release_key),
    default=None,
)
if isinstance(event, dict) and event.get("id") != st.session_state.last_event:
    st.session_state.last_event = event.get("id")
    st.session_state.form = event.get("form")
    st.session_state.result = None
    st.session_state.error = None
    if event.get("action") == "predict":
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
    st.rerun()
