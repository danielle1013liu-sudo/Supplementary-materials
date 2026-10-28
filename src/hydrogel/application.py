"""Inference, descriptor contributions and domain checks for the web application."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits

from .data import ROOT, load_c4, load_config, load_dataset, load_schema, load_split
from .data import numerical_descriptor

TARGET_LABELS = {"tnf": "TNF-alpha", "il10": "IL-10"}
AD_NEIGHBOURS = 5
AD_TOLERANCE = 1e-10
SHAP_TOLERANCE = 1e-7


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _checked_asset(directory: Path, manifest: dict, relative: str) -> Path:
    path = directory / relative
    expected = manifest["sha256"].get(relative)
    if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Deployment asset checksum does not match: {relative}")
    return path


def _feature_groups(schema: dict) -> tuple[list[str], dict[str, list[int]]]:
    names = list(schema["numeric"])
    for descriptor in schema["predictors"]:
        if descriptor in schema["categorical"]:
            names.extend(f"{descriptor}_{level}" for level in schema["categorical"][descriptor]["labels"])
    groups = {
        descriptor: [index for index, name in enumerate(names)
                     if name == descriptor or name.startswith(descriptor + "_")]
        for descriptor in schema["predictors"]
    }
    if len(names) != 35 or sorted(index for group in groups.values() for index in group) != list(range(35)):
        raise ValueError("The descriptor schema does not define the expected 35 model inputs.")
    return names, groups


def _verify_predictions(pipeline, reference: pd.DataFrame, target: str, family: str,
                        imputer: str, seed: int, dataset: pd.DataFrame, split: dict,
                        schema: dict, root: Path) -> None:
    indexed = dataset.set_index("record_id", drop=False)
    training = indexed.loc[split["train_ids"]]
    test = indexed.loc[split["test_ids"]]
    c4 = load_c4(target, root)
    frames = [training, test, c4]
    identities = [(scope, identity) for scope, frame in zip(["Training", "Test", "C4"], frames)
                  for identity in frame.record_id]
    expected = reference.loc[
        (reference.target == target) & (reference.family == family) & (reference.imputer == imputer)
        & (reference.model_seed == seed) & reference.scope.isin(["Training", "Test", "C4"])
    ].set_index(["scope", "record_id"])
    if not expected.index.is_unique or len(expected) != len(identities) or set(expected.index) != set(identities):
        raise ValueError(f"The {target} model reference does not cover its recorded populations.")
    expected = expected.reindex(pd.MultiIndex.from_tuples(identities, names=["scope", "record_id"]))
    features = pd.concat([frame[schema["predictors"]] for frame in frames], ignore_index=True)
    prediction = np.asarray(pipeline.predict(features), dtype=float)
    if not np.allclose(prediction, expected.predicted_pg_ml, rtol=0, atol=1e-7):
        raise ValueError(f"The {target} model does not reproduce its recorded predictions on this runtime.")


def _load_diagnostic(directory: Path, manifest: dict, target: str, training: pd.DataFrame,
                     train_ids: list[str], names: list[str], schema: dict) -> dict[str, Any]:
    prefix = f"diagnostics/{target}"
    paths = {name: _checked_asset(directory, manifest, f"{prefix}/{name}") for name in (
        "settings.json", "preprocessor.joblib", "reference.npz", "Training_reference_distances.csv",
    )}
    settings = _read_json(paths["settings.json"])
    if (settings["target"] != target or settings["Training_ids"] != train_ids
            or settings["Training_n"] != len(train_ids) or settings["transformed_names"] != names
            or settings["neighbors"] != AD_NEIGHBOURS or not settings["self_excluded"]
            or settings["outcomes_used"]):
        raise ValueError(f"The {target} domain settings do not match its Training population.")
    if hashlib.sha256(paths["preprocessor.joblib"].read_bytes()).hexdigest() != settings["preprocessor_sha256"]:
        raise ValueError(f"The {target} domain preprocessor differs from its recorded settings.")
    preprocessor = joblib.load(paths["preprocessor.joblib"])
    with np.load(paths["reference.npz"], allow_pickle=False) as reference_file:
        reference = reference_file["transformed_training"].copy()
        identifiers = reference_file["training_ids"].tolist()
        stored_distances = reference_file["self_excluded_mean5"].copy()
    if identifiers != train_ids or reference.shape != (len(train_ids), 35):
        raise ValueError(f"The {target} domain reference identities or dimensions have changed.")
    transformed = np.asarray(preprocessor.transform(training[schema["predictors"]]), dtype=float)
    if list(preprocessor.get_feature_names_out()) != names or not np.array_equal(transformed, reference):
        raise ValueError(f"The {target} domain preprocessor does not reproduce its reference coordinates.")
    distances = pairwise_distances(reference)
    np.fill_diagonal(distances, np.inf)
    mean5 = np.sort(distances, axis=1)[:, :AD_NEIGHBOURS].mean(axis=1)
    threshold = float(settings["threshold_q95"])
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"The {target} domain threshold must be positive and finite.")
    if (not np.allclose(mean5, stored_distances, rtol=0, atol=AD_TOLERANCE)
            or not np.isclose(np.quantile(mean5, .95, method="linear"), threshold,
                              rtol=0, atol=AD_TOLERANCE)):
        raise ValueError(f"The {target} stored domain distances or threshold could not be reproduced.")
    table = pd.read_csv(paths["Training_reference_distances.csv"], float_precision="round_trip")
    if (table.record_id.tolist() != train_ids
            or not np.allclose(table.self_excluded_mean5_distance, mean5, rtol=0, atol=AD_TOLERANCE)
            or not np.allclose(table.AD_ratio, mean5 / threshold, rtol=0, atol=AD_TOLERANCE)):
        raise ValueError(f"The {target} domain distance table does not match the reference population.")
    return {"preprocessor": preprocessor, "reference": reference, "threshold": threshold,
            "reference_ids": identifiers,
            "supported_levels": {name: set(training[name]) for name in schema["categorical"]}}


def load_resources(root: Path | str | None = None) -> dict[str, Any]:
    """Load and verify the saved Training models and domain diagnostics; fit nothing."""
    root = ROOT if root is None else Path(root).resolve()
    schema = load_schema(root)
    names, groups = _feature_groups(schema)
    directory = root / "deployment_assets"
    manifest = _read_json(directory / "manifest.json")
    reference_path = root / "reference/predictions.csv.gz"
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != manifest["reference_predictions_sha256"]:
        raise ValueError("The model prediction reference checksum does not match the deployment.")
    reference = pd.read_csv(reference_path, float_precision="round_trip")
    resources: dict[str, Any] = {"root": root, "schema": schema, "targets": {}}
    with threadpool_limits(limits=1):
        for target, expected_n in [("tnf", 120), ("il10", 104)]:
            config = load_config(target, root)
            split = load_split(target, root)
            train_ids = split["train_ids"]
            if len(train_ids) != expected_n or manifest["training_n"][target] != expected_n:
                raise ValueError(f"Unexpected {target} Training population size.")
            dataset = load_dataset(target, root)
            training = dataset.set_index("record_id").loc[train_ids]
            family = config["selected_model"]
            imputer = config["models"][family]["imputer"]
            model_manifest = _read_json(_checked_asset(directory, manifest, f"models/{target}/manifest.json"))
            if (model_manifest["target"] != target or model_manifest["family"] != family
                    or model_manifest["imputer"] != imputer or model_manifest["fit_scope"] != "Training"
                    or model_manifest["features"] != schema["predictors"]
                    or [item["seed"] for item in model_manifest["models"]] != config["model_seeds"]):
                raise ValueError(f"The {target} saved models do not match the selected Training configuration.")
            models = []
            for item in model_manifest["models"]:
                path = _checked_asset(directory, manifest, f"models/{target}/{item['path']}")
                if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                    raise ValueError(f"The {target} model bundle checksum does not match.")
                pipeline = joblib.load(path)
                if list(pipeline.named_steps["preprocess"].get_feature_names_out()) != names:
                    raise ValueError(f"The {target} predictive feature order differs from the schema.")
                _verify_predictions(pipeline, reference, target, family, imputer, item["seed"],
                                    dataset, split, schema, root)
                explainer = shap.TreeExplainer(pipeline.named_steps["model"], data=None,
                                               feature_perturbation="tree_path_dependent", model_output="raw")
                models.append({"seed": item["seed"], "pipeline": pipeline, "explainer": explainer})
            diagnostic = _load_diagnostic(directory, manifest, target, training, train_ids, names, schema)
            resources["targets"][target] = {
                "models": models, "family": family, "imputer": imputer, "training_n": expected_n,
                "model_seeds": list(config["model_seeds"]), "groups": groups, "diagnostic": diagnostic,
            }
    return resources


def form_frame(form: Mapping[str, Any], root: Path | str | None = None) -> pd.DataFrame:
    """Validate one formulation, retaining explicitly unreported numerical values."""
    schema = load_schema(root)
    missing = set(schema["predictors"]) - set(form)
    if missing:
        raise ValueError(f"Missing descriptors: {', '.join(sorted(missing))}.")
    row = {}
    for name in schema["predictors"]:
        value = form[name]
        if name in schema["numeric"]:
            row[name] = numerical_descriptor(value, name)
        else:
            if not isinstance(value, str) or value.strip() not in schema["categorical"][name]["labels"]:
                raise ValueError(f"Unknown category for {name}: {value!r}.")
            row[name] = value.strip()
    return pd.DataFrame([row], columns=schema["predictors"])


def _domain(frame: pd.DataFrame, diagnostic: dict, schema: dict) -> dict[str, Any]:
    transformed = np.asarray(diagnostic["preprocessor"].transform(frame), dtype=float)
    if not np.isfinite(transformed).all():
        raise ValueError("The formulation could not be represented by finite domain coordinates.")
    distances = pairwise_distances(transformed, diagnostic["reference"])[0]
    if not np.isfinite(distances).all():
        raise ValueError("The numerical descriptors exceed the supported calculation range.")
    indices = np.argsort(distances, kind="stable")[:AD_NEIGHBOURS]
    mean5 = float(distances[indices].mean())
    ratio = mean5 / diagnostic["threshold"]
    unsupported = [{"descriptor": name, "level": str(frame.iloc[0][name]),
                    "label": schema["categorical"][name]["labels"][frame.iloc[0][name]]}
                   for name, levels in diagnostic["supported_levels"].items()
                   if frame.iloc[0][name] not in levels]
    return {"mean5_distance": mean5, "threshold_q95": diagnostic["threshold"], "domain_ratio": ratio,
            "numerical_AD_pass": bool(ratio <= 1), "categorical_support_pass": not unsupported,
            "domain_pass": bool(ratio <= 1 and not unsupported), "unsupported_levels": unsupported,
            "nearest_neighbours": [{"record_id": diagnostic["reference_ids"][index],
                                    "distance": float(distances[index])} for index in indices]}


def _explain(frame: pd.DataFrame, resource: dict, schema: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions, grouped_values, expected_values = [], [], []
    for saved in resource["models"]:
        pipeline = saved["pipeline"]
        matrix = np.asarray(pipeline.named_steps["preprocess"].transform(frame), dtype=float)
        prediction = float(pipeline.predict(frame)[0])
        contributions = np.asarray(saved["explainer"].shap_values(matrix, approximate=False, check_additivity=True))
        base = np.asarray(saved["explainer"].expected_value).reshape(-1)
        if contributions.shape != (1, 35) or base.shape != (1,) or not np.isfinite(contributions).all():
            raise ValueError("The model returned an invalid descriptor explanation.")
        grouped = np.asarray([contributions[0, resource["groups"][name]].sum() for name in schema["predictors"]])
        if (not np.isfinite(prediction) or not np.isfinite(base[0])
                or abs(base[0] + contributions.sum() - prediction) > SHAP_TOLERANCE
                or abs(base[0] + grouped.sum() - prediction) > SHAP_TOLERANCE):
            raise ValueError("The descriptor contributions do not sum to the model prediction.")
        predictions.append(prediction)
        grouped_values.append(grouped)
        expected_values.append(float(base[0]))
    return np.asarray(predictions), np.asarray(grouped_values), np.asarray(expected_values)


def predict_form(form: Mapping[str, Any], resources: dict[str, Any]) -> dict[str, Any]:
    """Predict both cytokines with the saved models and their original preprocessing."""
    schema = resources["schema"]
    frame = form_frame(form, resources["root"])
    targets, warnings, prediction_rows, contribution_rows = [], [], [], []
    missing = [name for name in schema["numeric"] if pd.isna(frame.iloc[0][name])]
    if missing:
        warnings.append("Unreported numerical descriptors were imputed: " + ", ".join(missing) + ".")
    with threadpool_limits(limits=1):
        for target, resource in resources["targets"].items():
            domain = _domain(frame, resource["diagnostic"], schema)
            values, grouped, bases = _explain(frame, resource, schema)
            consensus = grouped.mean(axis=0)
            mean = float(values.mean())
            if abs(bases.mean() + consensus.sum() - mean) > SHAP_TOLERANCE:
                raise ValueError("The mean descriptor contributions do not sum to the mean prediction.")
            result = {
                "target": target, "mean": mean, "sd": float(values.std(ddof=1)), **domain,
                "invalid": bool(np.any(values < 0)), "expected_value_pg_ml": float(bases.mean()),
                "individual": [{"seed": saved["seed"], "value": float(value)}
                               for saved, value in zip(resource["models"], values)],
                "contributions": [{"descriptor": name, "SHAP_pg_ml": float(value)}
                                  for name, value in zip(schema["predictors"], consensus)],
            }
            targets.append(result)
            if not domain["domain_pass"]:
                warnings.append(f"{TARGET_LABELS[target]} is outside the applicability domain.")
            if domain["unsupported_levels"]:
                levels = ", ".join(f"{item['descriptor']}={item['level']}" for item in domain["unsupported_levels"])
                warnings.append(f"{TARGET_LABELS[target]} Training does not include these categorical levels: {levels}.")
            if result["invalid"]:
                warnings.append(f"{TARGET_LABELS[target]} includes a negative model prediction.")
            for individual in result["individual"]:
                prediction_rows.append({
                    "target": target, "model_seed": individual["seed"], "prediction_pg_ml": individual["value"],
                    "prediction_mean_pg_ml": mean, "prediction_sample_SD_pg_ml": result["sd"],
                    **{name: domain[name] for name in ["mean5_distance", "threshold_q95", "domain_ratio",
                                                      "numerical_AD_pass", "categorical_support_pass", "domain_pass"]},
                })
            contribution_rows.extend({"target": target, **item, "expected_value_pg_ml": result["expected_value_pg_ml"]}
                                     for item in result["contributions"])
    downloads = {
        "inputs_csv": frame.to_csv(index=False, float_format="%.17g"),
        "predictions_csv": pd.DataFrame(prediction_rows).to_csv(index=False, float_format="%.17g"),
        "contributions_csv": pd.DataFrame(contribution_rows).to_csv(index=False, float_format="%.17g"),
        "summary_csv": pd.DataFrame([
            {
                "target": TARGET_LABELS[item["target"]],
                "prediction_mean_pg_ml": item["mean"],
                "prediction_SD_pg_ml": item["sd"],
                "applicability_domain": "Within" if item["domain_pass"] else "Outside",
            }
            for item in targets
        ]).to_csv(index=False, float_format="%.17g"),
    }
    return {"targets": targets, "warnings": warnings, "downloads": downloads}
