"""Conditional non-dominance over the complete, recorded candidate library.

This module only filters stored profiles and predictions. It neither constructs
new formulations nor fits models. Every pairing uses the same eligible subset.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import ROOT, load_schema

EXPERIMENTAL = (
    "macrophage_model_code", "activation_stimulus_code",
    "exposure_mode_code", "exposure_time_h",
)
RANGES = ("Gprime_kPa", "pore_size_um")
OBJECTIVES = ("tnf_mean_pg_ml", "il10_mean_pg_ml")
CONSISTENCY_THRESHOLD = 6


@dataclass(frozen=True)
class CandidateLibrary:
    """Read-only, column-oriented representation of the complete recorded grid."""

    columns: Mapping[str, np.ndarray]
    manifest: dict[str, Any]
    schema: dict[str, Any]

    @property
    def version(self) -> str:
        return self.manifest["source_csv_sha256"]

    def __len__(self) -> int:
        return len(self.columns["candidate_id"])

    def frame(self, indices: np.ndarray, names: list[str]) -> pd.DataFrame:
        values = {}
        for name in names:
            array = self.columns[name][indices]
            values[name] = array.astype(str) if array.dtype.kind == "S" else array
        return pd.DataFrame(values)


@dataclass(frozen=True)
class ScreeningResult:
    constraints: dict[str, Any]
    matching_indices: np.ndarray
    mean_front_indices: np.ndarray
    retained_indices: np.ndarray
    retained_frequencies: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def library_manifest(root: Path | str | None = None) -> dict[str, Any]:
    root = ROOT if root is None else Path(root)
    path = root / "deployment_assets/screening/manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_library(root: Path | str | None = None) -> CandidateLibrary:
    """Verify and load required columns without loading provenance-only columns."""
    root = ROOT if root is None else Path(root)
    manifest = library_manifest(root)
    schema = load_schema(root)
    if (_sha256(root / "data/feature_schema.json") != manifest["schema_sha256"]
            or schema["predictors"] != manifest["features"]):
        raise ValueError("The candidate library descriptor schema does not match this application.")
    path = root / "deployment_assets/screening" / manifest["asset"]
    if _sha256(path) != manifest["asset_sha256"]:
        raise ValueError("The candidate library checksum does not match its manifest.")
    required = list(dict.fromkeys([
        "candidate_id", *manifest["features"], *OBJECTIVES,
        "tnf_sample_SD_pg_ml", "il10_sample_SD_pg_ml",
        "tnf_AD_ratio", "il10_AD_ratio",
        "tnf_all_categories_observed_in_Training", "il10_all_categories_observed_in_Training",
        "prediction_evaluated", "all_six_nonnegative", "eligible",
        *manifest["member_prediction_columns"]["tnf"],
        *manifest["member_prediction_columns"]["il10"],
    ]))
    columns = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in required:
            specification = manifest["columns"][name]
            array = archive[name]
            if specification["encoding"] == "dictionary":
                array = archive[name + "__levels"][array]
            if array.shape != (manifest["counts"]["generated_n"],):
                raise ValueError(f"Incomplete candidate column: {name}.")
            array.flags.writeable = False
            columns[name] = array
    library = CandidateLibrary(columns, manifest, schema)
    _verify_eligibility(library)
    return library


def _verify_eligibility(library: CandidateLibrary) -> None:
    columns = library.columns
    support = (columns["tnf_all_categories_observed_in_Training"]
               & columns["il10_all_categories_observed_in_Training"])
    domain = support & (columns["tnf_AD_ratio"] <= 1) & (columns["il10_AD_ratio"] <= 1)
    member_columns = sum(library.manifest["member_prediction_columns"].values(), [])
    nonnegative = domain.copy()
    for name in member_columns:
        values = columns[name]
        if not np.isfinite(values[domain]).all():
            raise ValueError("A recorded in-domain candidate prediction is not finite.")
        nonnegative &= values >= 0
    if (not np.array_equal(domain, columns["prediction_evaluated"])
            or not np.array_equal(nonnegative, columns["all_six_nonnegative"])
            or not np.array_equal(nonnegative, columns["eligible"])):
        raise ValueError("Recorded candidate eligibility differs from the original rule.")
    actual = {"generated_n": len(library), "predicted_n": int(domain.sum()),
              "eligible_n": int(nonnegative.sum())}
    if any(actual[name] != library.manifest["counts"][name] for name in actual):
        raise ValueError("The complete candidate population does not match the manifest.")


def constraint_options(library: CandidateLibrary) -> dict[str, Any]:
    """Return only category levels and exact numerical values present in the grid."""
    categorical = {}
    for name, definition in library.schema["categorical"].items():
        values = set(library.columns[name].astype(str))
        categorical[name] = [{"value": code, "label": label}
                             for code, label in definition["labels"].items() if code in values]
    numeric = {}
    for name in RANGES:
        values = np.unique(library.columns[name]).tolist()
        numeric[name] = {"min": values[0], "max": values[-1], "values": values}
    return {"categorical": categorical, "numeric": numeric,
            "exposure_times": np.unique(library.columns["exposure_time_h"]).tolist()}


def _number(value: Any, descriptor: str) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{descriptor} must be a finite non-negative number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{descriptor} must be a finite non-negative number.") from None
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"{descriptor} must be a finite non-negative number.")
    return number


def normalise_constraints(library: CandidateLibrary, constraints: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact restrictions; do not snap to the grid or relax a bound."""
    if not isinstance(constraints, Mapping):
        raise ValueError("Screening constraints must be a descriptor mapping.")
    unknown = set(constraints) - set(library.schema["predictors"])
    if unknown:
        raise ValueError("Unknown screening descriptors: " + ", ".join(sorted(unknown)))
    result = {}
    for name in library.schema["predictors"]:
        value = constraints.get(name)
        if name in library.schema["categorical"]:
            if value is None or value == "":
                result[name] = None
            elif not isinstance(value, str) or value not in library.schema["categorical"][name]["labels"]:
                raise ValueError(f"Unknown category for {name}: {value!r}.")
            else:
                result[name] = value
        elif name in RANGES:
            if value is None:
                result[name] = None
                continue
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError(f"{name} requires a lower and upper bound.")
            lower, upper = (_number(bound, name) for bound in value)
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"The lower bound for {name} exceeds its upper bound.")
            result[name] = None if lower is None and upper is None else [lower, upper]
        else:
            result[name] = _number(value, name)
    return result


def front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Minimise x and maximise y in O(n log n), preserving exact objective ties."""
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-dominance requires aligned, finite objective arrays.")
    result = np.zeros(len(x), dtype=bool)
    if not len(x):
        return result
    order = np.argsort(x, kind="stable")
    ordered_x, ordered_y = x[order], y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_x) != 0) + 1]
    maxima = np.maximum.reduceat(ordered_y, starts)
    previous = np.r_[-np.inf, np.maximum.accumulate(maxima)[:-1]]
    lengths = np.diff(np.r_[starts, len(order)])
    result[order] = ((ordered_y == np.repeat(maxima, lengths))
                     & np.repeat(maxima > previous, lengths))
    return result


def screen(library: CandidateLibrary, constraints: Mapping[str, Any]) -> ScreeningResult:
    """Recompute the mean and all nine pairing fronts in the matching population."""
    constraints = normalise_constraints(library, constraints)
    selected = library.columns["eligible"].copy()
    for name, value in constraints.items():
        if value is None:
            continue
        values = library.columns[name]
        if name in RANGES:
            lower, upper = value
            if lower is not None:
                selected &= values >= lower
            if upper is not None:
                selected &= values <= upper
        elif values.dtype.kind == "S":
            selected &= values == value.encode("utf-8")
        else:
            selected &= values == value
    indices = np.flatnonzero(selected)
    mean_front = front(*(library.columns[name][indices] for name in OBJECTIVES))
    frequency = np.zeros(len(indices), dtype=np.uint8)
    members = library.manifest["member_prediction_columns"]
    if len(members["tnf"]) != 3 or len(members["il10"]) != 3:
        raise ValueError("The library must contain the nine recorded target-model pairings.")
    for tnf in members["tnf"]:
        x = library.columns[tnf][indices]
        for il10 in members["il10"]:
            frequency += front(x, library.columns[il10][indices])
    retained = mean_front & (frequency >= CONSISTENCY_THRESHOLD)
    retained_indices = indices[retained]
    retained_frequencies = frequency[retained]
    order = np.lexsort((library.columns["candidate_id"][retained_indices],
                        -library.columns[OBJECTIVES[1]][retained_indices],
                        library.columns[OBJECTIVES[0]][retained_indices]))
    return ScreeningResult(constraints, indices, indices[mean_front],
                           retained_indices[order], retained_frequencies[order])


def candidate_form(library: CandidateLibrary, candidate_id: str) -> dict[str, Any]:
    """Return the exact ten stored inputs, without display rounding or imputation."""
    if not isinstance(candidate_id, str):
        raise ValueError("A candidate identifier must be a string.")
    ids = library.columns["candidate_id"]
    identity = candidate_id.encode("utf-8") if ids.dtype.kind == "S" else candidate_id
    indices = np.flatnonzero(ids == identity)
    if len(indices) != 1:
        raise ValueError("The candidate identifier is not in this library.")
    return library.frame(indices, library.schema["predictors"]).iloc[0].to_dict()


def result_payload(library: CandidateLibrary, result: ScreeningResult,
                   max_plot_points: int = 2000) -> dict[str, Any]:
    """Produce a compact display sample and complete retained-profile downloads."""
    if isinstance(max_plot_points, bool) or max_plot_points < 1:
        raise ValueError("The display sample size must be positive.")
    selected = result.retained_indices
    names = ["candidate_id", *library.schema["predictors"], *OBJECTIVES,
             "tnf_sample_SD_pg_ml", "il10_sample_SD_pg_ml", "tnf_AD_ratio", "il10_AD_ratio"]
    table = library.frame(selected, names)
    for target in ("tnf", "il10"):
        table[target + "_AD_pass"] = (
            (library.columns[target + "_AD_ratio"][selected] <= 1)
            & library.columns[target + "_all_categories_observed_in_Training"][selected])
    table["nondomination_frequency_of_9"] = result.retained_frequencies
    retained = []
    for row in table.to_dict("records"):
        retained.append({"candidate_id": row["candidate_id"],
                         "values": {name: row[name] for name in library.schema["predictors"]},
                         **{target + suffix: row[target + column]
                            for target in ("tnf", "il10")
                            for suffix, column in [("_mean", "_mean_pg_ml"), ("_sd", "_sample_SD_pg_ml"),
                                                   ("_AD_pass", "_AD_pass"), ("_AD_ratio", "_AD_ratio")]}})
    matching = result.matching_indices
    if len(matching) > max_plot_points:
        plotted = matching[np.linspace(0, len(matching) - 1, max_plot_points, dtype=int)]
    else:
        plotted = matching
    points = np.column_stack([library.columns[name][plotted] for name in OBJECTIVES]).tolist()
    retained_points = np.column_stack([library.columns[name][selected] for name in OBJECTIVES]).tolist()
    fixed = sum(result.constraints[name] is not None for name in EXPERIMENTAL)
    setting = "fully fixed" if fixed == len(EXPERIMENTAL) else "partially constrained" if fixed else "unrestricted"
    metadata = {"library_version": library.version, "constraints": result.constraints,
                "experimental_setting": setting, "matching_eligible_candidates": len(matching),
                "mean_front_profiles": len(result.mean_front_indices), "retained_profiles": len(selected),
                "objective_directions": {"tnf": "minimise", "il10": "maximise"},
                "consistency": "Mean-prediction front and at least six of nine recorded target-model pairing fronts, all within the same eligible subset.",
                "scope": "Conditional results from the existing candidate grid; distinct from the canonical global representatives."}
    table["library_version"] = library.version
    table["active_constraints_json"] = json.dumps(result.constraints, ensure_ascii=False, sort_keys=True)
    return {"library_version": library.version, "constraints": result.constraints,
            "experimental_setting": setting, "matching_count": len(matching),
            "mean_front_count": len(result.mean_front_indices), "retained_count": len(selected),
            "retained": retained,
            "plot": {"candidates": points, "retained": retained_points, "sampled": len(plotted) < len(matching)},
            "downloads": {"profiles_csv": table.to_csv(index=False, float_format="%.17g"),
                          "constraints_json": json.dumps(metadata, ensure_ascii=False, indent=2)}}
