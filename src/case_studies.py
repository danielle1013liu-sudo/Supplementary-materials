"""Explain the two paired case studies in Supplementary Table S7."""

import argparse
from pathlib import Path
import pickle
import zlib

import numpy as np
import pandas as pd

from .data import (ROOT, ID, RESPONSE, check_environment, check_inputs, feature_names,
                   load_data, load_cases, read_json, save_csv, save_json)
from .models import load_models
from .case_replay import run as replay_predictions
from .interpretation import (REFERENCE, check_references, check_shap_reference,
                             explain, settings)


def read_reference(name):
    return pd.read_csv(REFERENCE / name, float_precision="round_trip", keep_default_na=False, na_values=[""])


def run(output):
    output = Path(output)
    check_environment()
    check_inputs()
    check_references()
    replay_predictions(output / "verification")
    manifest = read_json(ROOT / "data/p15/study_exclusion.json")
    p15_reference = read_reference("P15_study_holdout_grouped_shap.csv")
    p15_delta_reference = read_reference("P15_study_holdout_delta_shap.csv")
    c4_observations = read_reference("C4_observations.csv")
    features = feature_names()
    summary, local, delta, inputs = [], [], [], []
    for study in ("P15", "C4"):
        for target in ("tnf", "il10"):
            if study == "P15":
                config = manifest["targets"][target]
                ids = config["pair_IDs"]
                frame = load_data(target).set_index(ID, drop=False).loc[ids].reset_index(drop=True)
                model_list = [(s, pickle.loads(zlib.decompress(
                    (ROOT / f"models/p15/{target}_{s}_P15_excluded.pklz").read_bytes()
                ))) for s in config["fixed_estimator_seeds"]]
                names = ["PEG microgel scaffold", "PEG microgel scaffold with tethered IL-10"]
                observed = frame[RESPONSE].to_numpy()
                fit_scope = "Separate P15 whole-study-excluded models"
                membership = "whole-study-excluded"
            else:
                frame = load_cases(target)
                ids = ["C4_Group2", "C4_Group4"]
                model_list = load_models(target)
                names = frame.formulation.tolist()
                observations = c4_observations[c4_observations.target.eq(target)].set_index(ID).loc[ids]
                observed = observations.observed_pg_ml.to_numpy()
                fit_scope = "Primary Training-only models"
                membership = "external"
            arrays = explain(model_list, frame[features], ids, [membership] * 2,
                             output / study, target)
            if study == "C4":
                check_shap_reference(target, "C4_EXTERNAL", arrays)
            changes = arrays["grouped"][:, 1] - arrays["grouped"][:, 0]
            prediction_changes = np.diff(arrays["predictions"], axis=1).ravel()
            np.testing.assert_allclose(changes.sum(axis=1), prediction_changes, atol=1e-7, rtol=0)
            for number, (realisation, _) in enumerate(model_list):
                if study == "P15":
                    ref = p15_reference[p15_reference.target.eq(target) & p15_reference.model_realisation.astype(str).eq(str(realisation))]
                    for position, row_id in enumerate(ids):
                        expected = ref[ref[ID].eq(row_id)].set_index("internal_field").loc[features]
                        np.testing.assert_allclose(arrays["grouped"][number, position], expected.grouped_SHAP_pg_ml, atol=1e-7, rtol=0)
                        np.testing.assert_allclose(arrays["expected_values"][number], expected.expected_value_pg_ml, atol=1e-10, rtol=0)
                    expected = p15_delta_reference[p15_delta_reference.target.eq(target) & p15_delta_reference.model_realisation.astype(str).eq(str(realisation))].set_index("internal_field").loc[features]
                    np.testing.assert_allclose(changes[number], expected.delta_SHAP_B_minus_A_pg_ml, atol=1e-7, rtol=0)
            for position, (row_id, formulation) in enumerate(zip(ids, names)):
                predicted = arrays["predictions"][:, position]
                summary.append({
                    "study": study, "target": target, ID: row_id, "formulation": formulation,
                    "fit_scope": fit_scope, "membership": membership,
                    "observed_pg_ml": observed[position],
                    "prediction_mean_pg_ml": predicted.mean(),
                    "prediction_sample_SD_pg_ml": predicted.std(ddof=1),
                    "absolute_error_pg_ml": abs(predicted.mean() - observed[position]),
                    "mean_individual_absolute_error_pg_ml": abs(predicted - observed[position]).mean(),
                })
                inputs.append({"study": study, "target": target, ID: row_id, "formulation": formulation,
                               **frame.iloc[position][features].to_dict()})
                for column, descriptor in enumerate(features):
                    local.append({
                        "study": study, "target": target, ID: row_id, "formulation": formulation,
                        "descriptor": descriptor,
                        "mean_SHAP_pg_ml": arrays["consensus"][position, column],
                        "sample_SD_SHAP_pg_ml": arrays["grouped"][:, position, column].std(ddof=1),
                        "expected_value_mean_pg_ml": arrays["expected_values"].mean(),
                        "prediction_mean_pg_ml": predicted.mean(),
                        "raw_value": frame.iloc[position][descriptor],
                        "is_missing": bool(pd.isna(frame.iloc[position][descriptor])),
                    })
            for column, descriptor in enumerate(features):
                first, second = frame.iloc[0][descriptor], frame.iloc[1][descriptor]
                unchanged = (pd.isna(first) and pd.isna(second)) or first == second
                delta.append({
                    "study": study, "target": target, "descriptor": descriptor,
                    "raw_descriptor_changed": not unchanged,
                    "mean_delta_SHAP_pg_ml": changes[:, column].mean(),
                    "sample_SD_delta_SHAP_pg_ml": changes[:, column].std(ddof=1),
                    "observed_change_pg_ml": observed[1] - observed[0],
                    "predicted_change_mean_pg_ml": prediction_changes.mean(),
                    "predicted_change_sample_SD_pg_ml": prediction_changes.std(ddof=1),
                })
    for name, rows in [("summary", summary), ("local_shap", local), ("delta_shap", delta), ("inputs", inputs)]:
        save_csv(pd.DataFrame(rows), output / f"{name}.csv")
    error_metrics = []
    for (study, target), rows in pd.DataFrame(summary).groupby(["study", "target"], sort=False):
        observed_change = rows.observed_pg_ml.iloc[1] - rows.observed_pg_ml.iloc[0]
        predicted_change = rows.prediction_mean_pg_ml.iloc[1] - rows.prediction_mean_pg_ml.iloc[0]
        error_metrics.append({
            "study": study, "target": target, "n_formulations": len(rows),
            "MAE_of_ensemble_mean_pg_ml": rows.absolute_error_pg_ml.mean(),
            "mean_individual_model_MAE_pg_ml": rows.mean_individual_absolute_error_pg_ml.mean(),
            "observed_pair_change_pg_ml": observed_change,
            "predicted_pair_change_pg_ml": predicted_change,
            "absolute_pair_change_error_pg_ml": abs(predicted_change - observed_change),
        })
    save_csv(pd.DataFrame(error_metrics), output / "error_metrics.csv")
    save_json({
        **settings(), "supplementary_table": "S7a,b",
        "P15": "Basal THP-1 at 72 h; whole-study-excluded objects; retrospective analysis; both formulations outside their corresponding applicability domains.",
        "C4": "HPA@SIN-GM versus HPA@SIN-GM@DR; the same primary objects used for Test prediction.",
        "interpretation": "Attributions describe model predictions. Nonzero delta-SHAP for an unchanged descriptor does not imply a changed experimental factor.",
        "reference_prediction_and_SHAP_checks_passed": True,
    }, output / "settings.json")
    check_inputs()
    check_references()
    return pd.DataFrame(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/case_studies")
    args = parser.parse_args()
    run(args.output)
    print("P15 and C4 predictions and local/delta SHAP reproduce the accepted case studies.")
