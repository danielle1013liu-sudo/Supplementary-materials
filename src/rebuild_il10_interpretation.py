"""Build the interpretation references for the approved IL-10 revision.

This is an explicit release-building operation, not the normal analysis replay.
It fits only the three P15 study-excluded sensitivity models. Primary models
must already pass the prediction locks produced by the independent seed screen.
Run from the repository root with ``python -m src.rebuild_il10_interpretation``.
"""

from datetime import datetime, timezone
from pathlib import Path
import pickle
import zlib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import (
    ROOT, ID, RESPONSE, check_environment, configuration, feature_names,
    file_hash, load_cases, load_data, read_csv, read_json, save_csv, save_json,
    split_data, split_definition, value_hash,
)
from .interpretation import REFERENCE, explain, save_explanation
from .models import load_models, make_pipeline
from .pareto.common import describe
from .tsne_diagnostics import caption_trustworthiness

SEEDS = [18571, 28094, 30787]
DATA_HASH = "87ec0a562440edb7a2138fba7f01385848241e2cec5c6872b69031be3bb54188"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def replace_target(path, rows):
    old = pd.read_csv(path, float_precision="round_trip", keep_default_na=False, na_values=[""])
    new = pd.DataFrame(rows)
    new = new.reindex(columns=old.columns)
    save_csv(pd.concat([old.loc[old.target.ne("il10")], new], ignore_index=True), path)


def check_primary():
    check_environment()
    assert configuration("il10")["seeds"] == SEEDS
    assert file_hash(ROOT / "data/il10.csv") == DATA_HASH
    models = load_models("il10")
    _, test = split_data("il10")
    cases = load_cases("il10")
    with np.load(ROOT / "data/reference/accepted_predictions.npz") as reference:
        for seed, model in models:
            assert model.named_steps["preprocess"].named_transformers_["numerical"].named_steps["imputer"].strategy == "median"
            for scope, values in [("Test", test), ("C4", cases)]:
                np.testing.assert_array_equal(
                    model.predict(values[feature_names()]), reference[f"il10_{seed}_{scope}"]
                )
    return models


def rebuild_primary_shap(models, output):
    frame = load_data("il10")
    split = split_definition("il10")
    membership = ["Training-in-sample" if row in split["Training_IDs"] else "Test-out-of-fit"
                  for row in frame[ID]]
    arrays = explain(models, frame[feature_names()], frame[ID].tolist(), membership,
                     output / "ALL_ROWS_DESCRIPTIVE", "il10")
    np.savez_compressed(REFERENCE / "il10_ALL_ROWS_DESCRIPTIVE_SHAP.npz", **arrays)
    index = [frame[ID].tolist().index(row) for row in split["Test_IDs"]]
    test = {
        "ids": np.asarray(split["Test_IDs"], dtype=str),
        "grouped": arrays["grouped"][:, index],
        "consensus": arrays["grouped"][:, index].mean(axis=0),
        "predictions": arrays["predictions"][:, index],
        "expected_values": arrays["expected_values"],
    }
    np.savez_compressed(REFERENCE / "il10_TEST_ONLY_SHAP.npz", **test)
    save_explanation(test, ["Test-out-of-fit"] * len(index), output / "TEST_ONLY", "il10")
    external = explain(models, load_cases("il10")[feature_names()],
                       ["C4_Group2", "C4_Group4"], ["external"] * 2,
                       output / "C4_EXTERNAL", "il10")
    np.savez_compressed(REFERENCE / "il10_C4_EXTERNAL_SHAP.npz", **external)


def rebuild_tsne(shap_output):
    descriptors = load_data("il10").set_index(ID)
    k10 = None
    for scope, perplexities in [("TEST_ONLY", [5, 10, 15]),
                               ("ALL_ROWS_DESCRIPTIVE", [10, 20, 30])]:
        # The normal runner reads this same exported CSV before embedding.
        frame = read_csv(shap_output / scope / "il10_consensus.csv")
        values = frame[feature_names()].to_numpy(float)
        for perplexity in perplexities:
            coordinates = TSNE(
                n_components=2, perplexity=perplexity, random_state=42,
                init="pca", learning_rate="auto", max_iter=2000,
                n_jobs=1, method="exact",
            ).fit_transform(values)
            out = frame[[ID, "membership", "prediction_mean_pg_ml"]].copy()
            out["tsne_1"], out["tsne_2"] = coordinates[:, 0], coordinates[:, 1]
            out = out.join(descriptors[feature_names()], on=ID, validate="one_to_one")
            path = REFERENCE / f"il10_{scope}_p{perplexity}.csv"
            save_csv(out, path)
            if scope == "ALL_ROWS_DESCRIPTIVE" and perplexity == 30:
                k10 = caption_trustworthiness(values, read_csv(path)[["tsne_1", "tsne_2"]])
    return float(k10)


def p15_domain(frame, cases, config):
    settings = configuration("il10")
    numeric, categorical = settings["numeric_columns"], settings["categorical_columns"]
    transform = ColumnTransformer([
        ("numeric", Pipeline([("median", SimpleImputer(strategy="median")),
                              ("standardise", StandardScaler())]), numeric),
        ("categorical", Pipeline([("missing", SimpleImputer(strategy="constant", fill_value="__missing__")),
                                  ("OHE", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         categorical),
    ], sparse_threshold=0)
    matrix = transform.fit_transform(frame[feature_names()])
    query = transform.transform(cases[feature_names()])
    reference_distances = pairwise_distances(matrix)
    np.fill_diagonal(reference_distances, np.inf)
    threshold = float(np.quantile(np.sort(reference_distances, axis=1)[:, :5].mean(axis=1),
                                  .95, method="linear"))
    distances = np.sort(pairwise_distances(query, matrix), axis=1)[:, :5].mean(axis=1)
    rows = []
    for index, (identity, row) in enumerate(cases.iterrows()):
        unseen = [f"{name}={row[name]}" for name in categorical
                  if pd.notna(row[name]) and row[name] not in set(frame[name].dropna())]
        rows.append({
            "target": "il10", ID: identity, "formulation": "AB"[index],
            "distance": distances[index], "study_excluded_threshold": threshold,
            "ratio": distances[index] / threshold, "inside": bool(distances[index] <= threshold),
            "unseen_categorical_levels": "; ".join(unseen),
            "reference_rows": len(frame), "reference_studies": config["retained_studies"],
        })
    config["AD_constructor_spec"] = describe(transform)
    config.pop("AD_independent_max_difference", None)
    config["AD_recomputed_with_corrected_training_categories"] = True
    path = ROOT / "models/p15/il10_P15_excluded_AD.pklz"
    path.write_bytes(zlib.compress(pickle.dumps(transform, protocol=5), 9))
    return rows, path


def rebuild_p15(output):
    path = ROOT / "data/p15/study_exclusion.json"
    manifest = read_json(path)
    config = manifest["targets"]["il10"]
    values = load_data("il10").set_index(ID, drop=False)
    identities = read_csv(ROOT / "data/record_identity.csv")
    excluded = identities.loc[
        identities.target.eq("il10") & identities.paper_id.eq("P15") & identities[ID].isin(values.index), ID
    ].tolist()
    assert set(excluded) == set(config["excluded_IDs"])
    assert set(config["fit_IDs_ordered"]) == set(values.index) - set(excluded)
    training = values.loc[config["fit_IDs_ordered"]]
    cases = values.loc[config["pair_IDs"]]
    assert cases.stimulation_context.eq("unstimulated_or_basal").all()
    assert cases.culture_or_exposure_duration_h.eq(72).all()
    model_records = read_json(ROOT / "models/manifest.json")
    fitted, records = [], []
    for seed in SEEDS:
        model = make_pipeline("il10", seed)
        started = utc_now()
        model.fit(training[feature_names()], training[RESPONSE])
        completed = utc_now()
        fitted.append((seed, model))
        model_path = ROOT / f"models/p15/il10_{seed}_P15_excluded.pklz"
        model_path.write_bytes(zlib.compress(pickle.dumps(model, protocol=5), 9))
        records.append({
            "target": "il10", "seed": seed, "file": model_path.relative_to(ROOT).as_posix(),
            "sha256": file_hash(model_path),
            "accepted_source_model_sha256": next(
                item["sha256"] for item in model_records
                if item["target"] == "il10" and item["seed"] == seed),
            "complete_constructor_spec": describe(model),
            "fit_started_utc": started, "fit_completed_utc": completed,
            "fit_IDs_sha256": value_hash(config["fit_IDs_ordered"]),
            "preprocessor_fitted_from_scratch": True,
        })
    config["fixed_estimator_seeds"] = SEEDS
    config["input_sha256"] = DATA_HASH
    manifest["fitted_models"] = [r for r in manifest["fitted_models"] if r["target"] != "il10"] + records
    manifest["data_release"] = read_json(ROOT / "data/feature_schema.json")["revision"]
    manifest["model_configuration_release"] = configuration().get("release", "IL10-crosslink7-RF-median-20261020")
    manifest["revision_note"] = (
        "IL-10 refitted after seven approved crosslink-level changes, with the accepted median RF "
        "parameters and three realisations. TNF fitted objects and predictions are unchanged."
    )
    arrays = explain(fitted, cases[feature_names()], config["pair_IDs"],
                     ["whole-study-excluded"] * 2, output / "P15", "il10")
    predictions, grouped, expected = arrays["predictions"], arrays["grouped"], arrays["expected_values"]
    observed = cases[RESPONSE].to_numpy()
    changes = np.diff(predictions, axis=1).ravel()
    labels = (pd.read_csv(REFERENCE / "P15_study_holdout_grouped_shap.csv")
              .drop_duplicates("internal_field").set_index("internal_field").feature.to_dict())
    pred_rows, paired_rows, shap_rows, delta_rows = [], [], [], []
    for index, seed in enumerate([*SEEDS, "ensemble_mean"]):
        ensemble = seed == "ensemble_mean"
        prediction = predictions.mean(axis=0) if ensemble else predictions[index]
        contributions = grouped.mean(axis=0) if ensemble else grouped[index]
        base = expected.mean() if ensemble else expected[index]
        delta = contributions[1] - contributions[0]
        for row, identity in enumerate(config["pair_IDs"]):
            error = prediction[row] - observed[row]
            pred_rows.append({
                "target": "il10", "model_realisation": seed, ID: identity,
                "formulation": "AB"[row], "observed_pg_ml": observed[row],
                "predicted_pg_ml": prediction[row],
                "prediction_sample_SD_pg_ml": predictions[:, row].std(ddof=1) if ensemble else np.nan,
                "error_predicted_minus_observed_pg_ml": error, "absolute_error_pg_ml": abs(error),
            })
            for col, feature in enumerate(feature_names()):
                shap_rows.append({
                    "target": "il10", "model_realisation": seed, ID: identity,
                    "formulation": "AB"[row], "feature": labels[feature], "internal_field": feature,
                    "grouped_SHAP_pg_ml": contributions[row, col],
                    "SHAP_sample_SD_pg_ml": grouped[:, row, col].std(ddof=1) if ensemble else np.nan,
                    "expected_value_pg_ml": base, "predicted_value_pg_ml": prediction[row],
                    "grouped_additivity_residual_pg_ml": base + contributions[row].sum() - prediction[row],
                })
        change = prediction[1] - prediction[0]
        observed_change = observed[1] - observed[0]
        paired_rows.append({
            "target": "il10", "model_realisation": seed,
            "observed_A_pg_ml": observed[0], "observed_B_pg_ml": observed[1],
            "predicted_A_pg_ml": prediction[0], "predicted_B_pg_ml": prediction[1],
            "observed_change_B_minus_A_pg_ml": observed_change,
            "predicted_change_B_minus_A_pg_ml": change,
            "paired_change_sample_SD_pg_ml": changes.std(ddof=1) if ensemble else np.nan,
            "paired_change_error_pg_ml": change - observed_change,
            "direction_matches_observed": bool(np.sign(change) == np.sign(observed_change)),
        })
        for col, feature in enumerate(feature_names()):
            first, second = cases.iloc[0][feature], cases.iloc[1][feature]
            unchanged = (pd.isna(first) and pd.isna(second)) or first == second
            delta_rows.append({
                "target": "il10", "model_realisation": seed, "feature": labels[feature],
                "internal_field": feature, "delta_SHAP_B_minus_A_pg_ml": delta[col],
                "delta_SHAP_sample_SD_pg_ml": (grouped[:, 1, col] - grouped[:, 0, col]).std(ddof=1) if ensemble else np.nan,
                "expected_value_difference_pg_ml": 0,
                "predicted_change_B_minus_A_pg_ml": change, "sum_residual_pg_ml": delta.sum() - change,
                "raw_descriptor_changed": not unchanged,
            })
    for name, rows in [("study_holdout_predictions.csv", pred_rows),
                       ("study_holdout_paired_changes.csv", paired_rows)]:
        replace_target(ROOT / "data/p15" / name, rows)
    for name, rows in [("P15_study_holdout_grouped_shap.csv", shap_rows),
                       ("P15_study_holdout_delta_shap.csv", delta_rows)]:
        replace_target(REFERENCE / name, rows)
    ad_rows, ad_path = p15_domain(training, cases, config)
    replace_target(ROOT / "data/p15/study_holdout_AD.csv", ad_rows)
    save_json(manifest, path)
    hashes = read_json(ROOT / "data/p15/checksums.json")
    hashes = {name: file_hash(ROOT / name) for name in hashes if not name.startswith("models/p15/il10_")}
    for record in records:
        hashes[record["file"]] = record["sha256"]
    hashes[ad_path.relative_to(ROOT).as_posix()] = file_hash(ad_path)
    save_json(hashes, ROOT / "data/p15/checksums.json")
    return {"prediction_mean": predictions.mean(axis=0).tolist(),
            "prediction_SD": predictions.std(axis=0, ddof=1).tolist(),
            "AD": ad_rows, "model_records": records}


def main():
    models = check_primary()
    output = ROOT / "outputs/release_build/il10_20261020"
    output.mkdir(parents=True, exist_ok=True)
    preserved = {p.relative_to(ROOT).as_posix(): file_hash(p)
                 for pattern in ["models/tnf_*.pklz", "models/p15/tnf_*.pklz",
                                 "data/interpretation/reference/tnf_*"]
                 for p in ROOT.glob(pattern)}
    previous_reference_hashes = read_json(REFERENCE.parent / "checksums.json")
    rebuild_primary_shap(models, output / "shap")
    k10 = rebuild_tsne(output / "shap")
    p15 = rebuild_p15(output)
    provenance_path = REFERENCE.parent / "provenance.json"
    provenance = read_json(provenance_path)
    provenance["model_release"] = configuration().get("release", "IL10-crosslink7-RF-median-20261020")
    provenance["data_release"] = read_json(ROOT / "data/feature_schema.json")["revision"]
    provenance["supplement"] = "Supplementary_IL10_updated_20261020.docx"
    provenance["primary_tSNE_trustworthiness_k10"]["il10"] = k10
    provenance["revision"] = {
        "created_utc": utc_now(), "builder": "src/rebuild_il10_interpretation.py",
        "primary_predictive_fits": 0, "P15_predictive_fits": 3,
        "P15_diagnostic_preprocessing_fits": 1, "primary_data_sha256": DATA_HASH,
        "previous_reference_checksums": previous_reference_hashes,
        "TNF_reference_and_model_bytes_unchanged": True,
    }
    for item in provenance["sources"]:
        if item["file"].startswith(("il10_", "P15_")):
            item["source"] = "src/rebuild_il10_interpretation.py; TNF portions preserved where shared"
    save_json(provenance, provenance_path)
    hashes = read_json(REFERENCE.parent / "checksums.json")
    save_json({name: file_hash(ROOT / name) for name in hashes},
              REFERENCE.parent / "checksums.json")
    for name, expected in preserved.items():
        assert file_hash(ROOT / name) == expected, name
    save_json({"built_utc": utc_now(), "tnf_unchanged": preserved,
               "il10_tSNE_trustworthiness_k10": k10, "P15": p15},
              output / "build_report.json")
    print(f"IL-10 interpretation references rebuilt; t-SNE k10={k10:.15f}", flush=True)


if __name__ == "__main__":
    main()
