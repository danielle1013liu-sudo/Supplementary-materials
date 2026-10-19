"""Shared TreeSHAP calculation and reference checks for the accepted models."""

from pathlib import Path

import numpy as np
import pandas as pd
import shap

from .data import ROOT, ID, feature_names, file_hash, read_json, save_csv, save_json

REFERENCE = ROOT / "data/interpretation/reference"


def check_references():
    for name, expected in read_json(REFERENCE.parent / "checksums.json").items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Interpretation reference changed: {name}")


def explain(models, values, ids, membership, destination, target):
    """Sum signed one-hot contributions within each original descriptor."""
    features = feature_names()
    grouped, predictions, expected, checks = [], [], [], []
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for realisation, pipeline in models:
        transform = pipeline.named_steps["preprocess"]
        matrix = np.asarray(transform.transform(values), dtype=float)
        names = transform.get_feature_names_out().tolist()
        mapping = []
        for name in names:
            matches = [f for f in features if name == f or name.startswith(f + "_")]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous transformed descriptor: {name}")
            mapping.append(features.index(matches[0]))
        explainer = shap.TreeExplainer(
            pipeline.named_steps["model"], data=None,
            feature_perturbation="tree_path_dependent", model_output="raw",
        )
        contributions = np.asarray(explainer.shap_values(
            matrix, approximate=False, check_additivity=True,
        ))
        base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        predicted = pipeline.predict(values)
        combined = np.zeros((len(values), len(features)))
        for column, feature in enumerate(mapping):
            combined[:, feature] += contributions[:, column]
        np.testing.assert_allclose(base + combined.sum(axis=1), predicted, rtol=0, atol=1e-7)
        np.testing.assert_allclose(base + contributions.sum(axis=1), predicted, rtol=0, atol=1e-7)
        metadata = pd.DataFrame({
            ID: ids, "membership": membership,
            "expected_value_pg_ml": base, "prediction_pg_ml": predicted,
        })
        for level, array, columns in [("transformed", contributions, names), ("grouped", combined, features)]:
            save_csv(pd.concat([metadata, pd.DataFrame(array, columns=columns)], axis=1),
                     destination / f"{target}_{realisation}_{level}.csv")
        grouped.append(combined)
        predictions.append(predicted)
        expected.append(base)
        checks.append({
            "target": target, "model_realisation": realisation, "n": len(values),
            "max_additivity_residual_pg_ml": float(abs(base + combined.sum(axis=1) - predicted).max()),
        })
    arrays = {
        "grouped": np.asarray(grouped), "predictions": np.asarray(predictions),
        "expected_values": np.asarray(expected), "ids": np.asarray(ids, dtype=str),
    }
    arrays["consensus"] = arrays["grouped"].mean(axis=0)
    save_explanation(arrays, membership, destination, target)
    save_json(checks, destination / f"{target}_additivity.json")
    return arrays


def save_explanation(arrays, membership, destination, target):
    destination = Path(destination)
    metadata = pd.DataFrame({
        ID: arrays["ids"], "membership": membership,
        "prediction_mean_pg_ml": arrays["predictions"].mean(axis=0),
        "prediction_sample_SD_pg_ml": arrays["predictions"].std(axis=0, ddof=1),
        "expected_value_mean_pg_ml": arrays["expected_values"].mean(),
    })
    save_csv(pd.concat([metadata, pd.DataFrame(arrays["consensus"], columns=feature_names())], axis=1),
             destination / f"{target}_consensus.csv")
    np.savez_compressed(destination / f"{target}_arrays.npz", **arrays)


def check_shap_reference(target, scope, arrays):
    # Older frozen archives store row identifiers as NumPy object arrays.
    with np.load(REFERENCE / f"{target}_{scope}_SHAP.npz", allow_pickle=True) as expected:
        for key in ("ids", "grouped", "consensus", "predictions", "expected_values"):
            np.testing.assert_array_equal(arrays[key], expected[key], err_msg=f"{target}/{scope}/{key}")


def settings():
    return {
        "feature_perturbation": "tree_path_dependent", "model_output": "raw",
        "background": None, "approximate": False,
        "grouping": "Signed transformed-column sums within the ten canonical descriptors",
        "ensemble": "Arithmetic mean of signed contributions across three saved models",
        "uncertainty": "Sample SD across the three models (ddof=1)",
        "predictive_fits": 0, "preprocessing_fits": 0,
    }
