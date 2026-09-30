"""Signed grouped TreeSHAP for the selected primary-model profiles."""

from .common import FEATURES, Path, SEEDS, csv, dump, json, models, np, pd, read, table
import shap


def run(output):
    """Write this numerical stage below the separate output directory."""
    output = Path(output)
    out = output / "results"
    representatives = csv(out / "FINAL_PARETO_REPRESENTATIVES.csv")
    assert representatives.display_label.tolist() == ["P1", "P2", "P3"]
    grouped_rows, individual_rows, residuals, mappings = [], [], [], []
    for target in SEEDS:
        arrays, bases, predictions = [], [], []
        for seed, pipe in models(target):
            pp = pipe.named_steps["preprocess"]
            z = np.asarray(pp.transform(representatives[FEATURES]), float)
            names = pp.get_feature_names_out().tolist()
            fields = []
            for name in names:
                matches = [f for f in FEATURES if name == f or name.startswith(f + "_")]
                assert len(matches) == 1
                fields.append(matches[0])
                mappings.append(
                    {
                        "target": target,
                        "realisation": seed,
                        "transformed_feature": name,
                        "descriptor_name": matches[0],
                    }
                )
            explainer = shap.TreeExplainer(
                pipe.named_steps["model"],
                data=None,
                feature_perturbation="tree_path_dependent",
                model_output="raw",
            )
            values = np.asarray(
                explainer.shap_values(z, approximate=False, check_additivity=True)
            )
            base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
            prediction = pipe.predict(representatives[FEATURES])
            np.testing.assert_array_equal(
                prediction, representatives[f"{target}_seed_{seed}_pg_ml"].to_numpy()
            )
            grouped = np.zeros((len(representatives), len(FEATURES)))
            for j, field in enumerate(fields):
                grouped[:, FEATURES.index(field)] += values[:, j]
            np.testing.assert_allclose(
                base + grouped.sum(axis=1), prediction, rtol=1e-10, atol=1e-7
            )
            for i, rep in representatives.iterrows():
                residuals.append(
                    {
                        "target": target,
                        "candidate_id": rep.candidate_id,
                        "model_realisation": str(seed),
                        "expected_value_pg_ml": base,
                        "prediction_pg_ml": prediction[i],
                        "grouped_sum_pg_ml": grouped[i].sum(),
                        "additivity_residual_pg_ml": base
                        + grouped[i].sum()
                        - prediction[i],
                    }
                )
                for j, name in enumerate(names):
                    individual_rows.append(
                        {
                            "target": target,
                            "candidate_id": rep.candidate_id,
                            "display_label": rep.display_label,
                            "model_realisation": seed,
                            "transformed_feature": name,
                            "descriptor_name": fields[j],
                            "transformed_value": z[i, j],
                            "SHAP_pg_ml": values[i, j],
                            "expected_value_pg_ml": base,
                            "prediction_pg_ml": prediction[i],
                        }
                    )
                for j, field in enumerate(FEATURES):
                    grouped_rows.append(
                        {
                            "target": target,
                            "candidate_id": rep.candidate_id,
                            "display_label": rep.display_label,
                            "model_realisation": str(seed),
                            "descriptor_name": field,
                            "grouped_SHAP_pg_ml": grouped[i, j],
                            "expected_value_pg_ml": base,
                            "prediction_pg_ml": prediction[i],
                        }
                    )
            arrays.append(grouped)
            bases.append(base)
            predictions.append(prediction)
        mean_values = np.mean(arrays, axis=0)
        mean_base = float(np.mean(bases))
        mean_prediction = np.mean(predictions, axis=0)
        np.testing.assert_allclose(
            mean_prediction, representatives[target + "_mean_pg_ml"], atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(
            mean_base + mean_values.sum(axis=1), mean_prediction, atol=1e-7, rtol=1e-10
        )
        representatives[target + "_expected_value_pg_ml"] = mean_base
        for i, rep in representatives.iterrows():
            residuals.append(
                {
                    "target": target,
                    "candidate_id": rep.candidate_id,
                    "model_realisation": "mean",
                    "expected_value_pg_ml": mean_base,
                    "prediction_pg_ml": mean_prediction[i],
                    "grouped_sum_pg_ml": mean_values[i].sum(),
                    "additivity_residual_pg_ml": mean_base
                    + mean_values[i].sum()
                    - mean_prediction[i],
                }
            )
            for j, field in enumerate(FEATURES):
                grouped_rows.append(
                    {
                        "target": target,
                        "candidate_id": rep.candidate_id,
                        "display_label": rep.display_label,
                        "model_realisation": "mean",
                        "descriptor_name": field,
                        "grouped_SHAP_pg_ml": mean_values[i, j],
                        "expected_value_pg_ml": mean_base,
                        "prediction_pg_ml": mean_prediction[i],
                    }
                )
    table(out / "FINAL_PARETO_REPRESENTATIVES.csv", representatives)
    table(out / "REPRESENTATIVE_SHAP_ALL_REALISATIONS.csv", grouped_rows)
    table(out / "REPRESENTATIVE_SHAP_TRANSFORMED.csv", individual_rows)
    table(
        out / "FINAL_REPRESENTATIVE_SHAP.csv",
        pd.DataFrame(grouped_rows).query('model_realisation == "mean"'),
    )
    table(output / "checks/SHAP_ADDITIVITY.csv", residuals)
    table(out / "TRANSFORMED_TO_DESCRIPTOR_MAP.csv", mappings)
    dump(
        output / "checks/SHAP_CONVENTION.json",
        {
            "feature_perturbation": "tree_path_dependent",
            "model_output": "raw",
            "background": "Training path counts saved in the primary trees; no new background",
            "approximate": False,
            "check_additivity": True,
            "grouping": "Signed sum of transformed-feature contributions per canonical descriptor",
            "model_average": "Arithmetic mean over the three unchanged saved models per target",
            "max_absolute_residual_pg_ml": max(
                abs(r["additivity_residual_pg_ml"]) for r in residuals
            ),
            "model_fits": 0,
        },
    )
    print(json.dumps(read(output / "checks/SHAP_CONVENTION.json"), indent=2))
