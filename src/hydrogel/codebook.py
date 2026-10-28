"""A concise descriptor guide derived from the verified category mappings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data import load_schema


DESCRIPTORS = {
    "hydrogel_composition_code": (
        "Hydrogel composition", None,
        "Polymer composition of the hydrogel network: natural, synthetic or a natural–synthetic hybrid.",
    ),
    "Gprime_kPa": (
        "Storage modulus G′", "kPa",
        "Measured shear storage modulus of the hydrogel, expressed in kilopascals.",
    ),
    "pore_size_um": (
        "Pore size", "µm",
        "Reported pore dimension in micrometres; this is not porosity or theoretical mesh size.",
    ),
    "bioactive_cue_code": (
        "Bioactive cue", None,
        "Functional class of incorporated bioactive cues. Repair-directed cell recruitment belongs to "
        "pro-regenerative signalling; combined anti-inflammatory and pro-regenerative cues use the combined category.",
    ),
    "crosslink_method_code": (
        "Crosslinking method", None,
        "Method used to form the hydrogel network. Mixed or other methods use the corresponding category.",
    ),
    "crosslink_level_code": (
        "Crosslinking level", None,
        "Reported relative crosslinking level, represented as low, medium or high. These categories "
        "do not specify a universal chemical concentration or crosslink-density threshold.",
    ),
    "macrophage_model_code": (
        "Macrophage model", None,
        "Macrophage cell line, primary-cell source or experimental biological context.",
    ),
    "activation_stimulus_code": (
        "Activation stimulus", None,
        "Externally applied macrophage-activating or polarising stimulus. Basal means no externally "
        "applied polarising stimulus, independently of the hydrogel's bioactive cue.",
    ),
    "exposure_mode_code": (
        "Exposure mode", None,
        "How macrophages encounter the material, its extract or eluate, or an in vivo exposure.",
    ),
    "exposure_time_h": (
        "Exposure time", "h",
        "Duration of the recorded culture or material exposure, expressed in hours.",
    ),
}


def descriptor_codebook(root: Path | str | None = None) -> dict[str, Any]:
    """Return only user-facing definitions and the current category code mappings."""
    schema = load_schema(root)
    if set(schema["predictors"]) != set(DESCRIPTORS):
        raise ValueError("The descriptor guide does not cover the current feature schema.")
    descriptors = []
    for column in schema["predictors"]:
        name, unit, definition = DESCRIPTORS[column]
        row = {
            "column": column,
            "name": name,
            "type": "numerical" if column in schema["numeric"] else "categorical",
            "unit": unit,
            "definition": definition,
        }
        if column in schema["categorical"]:
            row["categories"] = [
                {"code": code, "label": label}
                for code, label in schema["categorical"][column]["labels"].items()
            ]
        descriptors.append(row)
    return {
        "title": "Hydrogel Cytokine Explorer descriptor guide",
        "descriptors": descriptors,
        "missing_values": {
            "numerical": "Select Unreported in the prediction form or leave the CSV cell empty. "
            "Explicitly missing measurements remain missing until the fitted preprocessing pipeline processes them. "
            "Present values must be finite and non-negative; do not substitute zero for an unreported measurement.",
            "categorical": "Use one of the listed category codes. Missing and unknown category codes are not accepted.",
        },
    }


def codebook_json(root: Path | str | None = None) -> str:
    """Serialise the public guide without exposing historical workflow metadata."""
    return json.dumps(descriptor_codebook(root), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
