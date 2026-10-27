"""Load the released datasets, descriptor schema and ordered partitions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
TARGET_SIZES = {"tnf": 150, "il10": 131}
RECORD_ID = "record_id"
RESPONSE = "cytokine_pg_ml"
SOURCE_DOI = "source_doi"


def _target(target: str) -> str:
    value = target.lower()
    if value not in TARGET_SIZES:
        raise ValueError(f"Unknown target: {target!r}; expected 'tnf' or 'il10'.")
    return value


def _root(root: Path | str | None) -> Path:
    return ROOT if root is None else Path(root).resolve()


def _checked_path(relative: str, root: Path | str | None) -> Path:
    base = _root(root)
    manifest = json.loads((base / "data/manifest.json").read_text(encoding="utf-8-sig"))
    expected = manifest["sha256"].get(relative)
    if expected is None:
        raise ValueError(f"File is absent from the data manifest: {relative}")
    path = base / relative
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"File checksum does not match the release: {relative}")
    return path


def _read_json(relative: str, root: Path | str | None) -> dict[str, Any]:
    return json.loads(_checked_path(relative, root).read_text(encoding="utf-8-sig"))


def load_schema(root: Path | str | None = None) -> dict[str, Any]:
    return _read_json("data/feature_schema.json", root)


def load_config(target: str, root: Path | str | None = None) -> dict[str, Any]:
    target = _target(target)
    config = _read_json(f"config/{target}.json", root)
    if config["target"] != target:
        raise ValueError("Configuration target does not match the requested dataset.")
    return config


def load_p32_scope(target: str, root: Path | str | None = None) -> dict[str, Any]:
    """Return the checksum-verified P32 fitting and case memberships."""
    target = _target(target)
    scope = _read_json(f"data/splits/{target}_p32.json", root)
    if scope["target"] != target:
        raise ValueError("P32 scope target does not match the requested dataset.")
    return scope


def validate_features(frame: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    """Validate raw descriptors without filling missing numerical measurements."""
    missing = set(schema["predictors"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing descriptor columns: {sorted(missing)}")
    result = frame.copy()
    for name in schema["numeric"]:
        result[name] = pd.to_numeric(result[name], errors="raise")
        if not np.isfinite(result[name].dropna()).all():
            raise ValueError(f"Non-finite numerical descriptor: {name}")
    for name in schema["predictors"]:
        if name not in schema["categorical"]:
            continue
        allowed = set(schema["categorical"][name]["labels"])
        invalid = ~result[name].isin(allowed)
        if invalid.any():
            values = result.loc[invalid, name].unique().tolist()
            raise ValueError(f"Unknown or missing categories in {name}: {values}")
    return result


def load_dataset(target: str, root: Path | str | None = None) -> pd.DataFrame:
    target = _target(target)
    path = _checked_path(f"data/{target}.csv", root)
    frame = pd.read_csv(path, keep_default_na=False, na_values=["", "\u2014"])
    frame = validate_features(frame, load_schema(root))
    frame[RESPONSE] = pd.to_numeric(frame[RESPONSE], errors="raise")
    if not np.isfinite(frame[RESPONSE]).all():
        raise ValueError("Responses must be finite numerical values.")
    if frame[RECORD_ID].isna().any() or not frame[RECORD_ID].is_unique:
        raise ValueError("Record identifiers must be present and unique.")
    if len(frame) != TARGET_SIZES[target]:
        raise ValueError(f"Unexpected number of observations for {target}.")
    return frame


def load_split(target: str, root: Path | str | None = None) -> dict[str, Any]:
    """Return the recorded row order and verify its split and fold assignments."""
    target = _target(target)
    split = _read_json(f"data/splits/{target}.json", root)
    config = load_config(target, root)
    frame = load_dataset(target, root)
    for key in ("split_seed", "cv_seed"):
        if split[key] != config[key]:
            raise ValueError(f"Configuration and recorded partition disagree on {key}.")
    training, test = train_test_split(
        np.arange(len(frame)),
        test_size=config["test_size"],
        shuffle=True,
        random_state=split["split_seed"],
    )
    if split["train_ids"] != frame.iloc[training][RECORD_ID].tolist():
        raise ValueError("Recorded Training membership or row order is inconsistent.")
    if split["test_ids"] != frame.iloc[test][RECORD_ID].tolist():
        raise ValueError("Recorded Test membership or row order is inconsistent.")
    folds = KFold(config["n_splits"], shuffle=True, random_state=split["cv_seed"])
    expected_labels = np.zeros(len(training), dtype=int)
    for label, (_, validation) in enumerate(folds.split(training), start=1):
        expected_labels[validation] = label
    if split["fold_labels"] != expected_labels.tolist():
        raise ValueError("Recorded cross-validation fold assignments are inconsistent.")
    return split


def load_c4(target: str, root: Path | str | None = None) -> pd.DataFrame:
    target = _target(target)
    path = _checked_path("data/cases/c4.csv", root)
    frame = pd.read_csv(path, keep_default_na=False, na_values=["", "\u2014"])
    cases = validate_features(frame.loc[frame["target"].eq(target)], load_schema(root))
    case_ids = ["C4_Group2", "C4_Group4"]
    if len(cases) != 2 or set(cases["case_id"]) != set(case_ids):
        raise ValueError("C4 requires exactly the Group2 and Group4 formulations.")
    cases = cases.set_index("case_id", drop=False).loc[case_ids].reset_index(drop=True)
    cases[RECORD_ID] = cases["case_id"]
    cases["observed_pg_ml"] = pd.to_numeric(cases["observed_pg_ml"], errors="raise")
    if not np.isfinite(cases["observed_pg_ml"]).all():
        raise ValueError("C4 observations must be finite numerical values.")
    return cases
