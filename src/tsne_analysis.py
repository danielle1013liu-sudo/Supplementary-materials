"""Reproduce the saved SHAP embeddings and their neighbourhood diagnostics."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors

from .data import (ROOT, ID, check_environment, check_inputs, feature_names,
                   load_data, read_csv, read_json, save_csv, save_json, split_definition)
from .interpretation import REFERENCE, check_references
from .tsne_diagnostics import caption_trustworthiness


def neighbours(values):
    indices = NearestNeighbors(n_neighbors=len(values)).fit(values).kneighbors(values)[1]
    return [set(row[row != index][:5]) for index, row in enumerate(indices)]


def run(shap_dir, output):
    shap_dir, output = Path(shap_dir), Path(output)
    check_environment()
    check_inputs()
    check_references()
    diagnostics, stability = [], []
    for target in ("tnf", "il10"):
        descriptors = load_data(target).set_index(ID)
        for scope, perplexities in [("TEST_ONLY", [5, 10, 15]), ("ALL_ROWS_DESCRIPTIVE", [10, 20, 30])]:
            # Preserve the CSV parsing used to obtain the archived embeddings.
            frame = read_csv(shap_dir / scope / f"{target}_consensus.csv")
            with np.load(REFERENCE / f"{target}_{scope}_SHAP.npz", allow_pickle=True) as reference:
                np.testing.assert_array_equal(frame[ID], reference["ids"])
                np.testing.assert_allclose(frame[feature_names()], reference["consensus"], atol=1e-10, rtol=0)
                np.testing.assert_allclose(frame.prediction_mean_pg_ml, reference["predictions"].mean(axis=0), atol=1e-10, rtol=0)
            training_ids = set(split_definition(target)["Training_IDs"])
            expected_membership = ["Training-in-sample" if row in training_ids else "Test-out-of-fit" for row in frame[ID]]
            np.testing.assert_array_equal(frame.membership, expected_membership)
            if not frame[ID].is_unique:
                raise ValueError("An embedding must have one row per observation")
            values = frame[feature_names()].to_numpy(float)
            original_neighbours = neighbours(values)
            projected = {}
            for perplexity in perplexities:
                embedding = TSNE(
                    n_components=2, perplexity=perplexity, random_state=42,
                    init="pca", learning_rate="auto", max_iter=2000,
                    n_jobs=1, method="exact",
                )
                coordinates = embedding.fit_transform(values)
                # Saved coordinates have NumPy float32 CSV formatting. Comparing
                # after that serialization avoids a float32/float64 display gap.
                coordinate_frame = frame[[ID, "membership", "prediction_mean_pg_ml"]].copy()
                coordinate_frame["tsne_1"] = coordinates[:, 0]
                coordinate_frame["tsne_2"] = coordinates[:, 1]
                path = output / scope / f"{target}_p{perplexity}.csv"
                coordinate_frame = coordinate_frame.join(descriptors[feature_names()], on=ID, validate="one_to_one")
                save_csv(coordinate_frame, path)
                actual = read_csv(path)
                reference = read_csv(REFERENCE / f"{target}_{scope}_p{perplexity}.csv")
                np.testing.assert_array_equal(actual[ID], reference[ID])
                np.testing.assert_array_equal(actual[["tsne_1", "tsne_2"]], reference[["tsne_1", "tsne_2"]])
                projected[perplexity] = neighbours(coordinates)
                if scope == "ALL_ROWS_DESCRIPTIVE" and perplexity == 30:
                    k10 = caption_trustworthiness(values, actual[["tsne_1", "tsne_2"]])
                    expected = read_json(REFERENCE.parent / "provenance.json")["primary_tSNE_trustworthiness_k10"][target]
                    np.testing.assert_allclose(k10, expected, atol=1e-15, rtol=0)
                else:
                    k10 = trustworthiness(values, actual[["tsne_1", "tsne_2"]], n_neighbors=10)
                diagnostics.append({
                    "target": target, "scope": scope, "n": len(values), "perplexity": perplexity,
                    "actual_iterations": embedding.n_iter_ + 1,
                    "KL_divergence": embedding.kl_divergence_,
                    "trustworthiness_k5": trustworthiness(values, coordinates, n_neighbors=5),
                    "trustworthiness_k10": k10,
                    "mean_neighbour_overlap_k5": float(np.mean([len(a & b) / 5 for a, b in zip(original_neighbours, projected[perplexity])])),
                    "exact_coordinate_reproduction": True,
                })
            for index, first in enumerate(perplexities):
                for second in perplexities[index + 1:]:
                    stability.append({
                        "target": target, "scope": scope, "perplexity_1": first, "perplexity_2": second,
                        "mean_neighbour_Jaccard_k5": float(np.mean([len(a & b) / len(a | b) for a, b in zip(projected[first], projected[second])])),
                    })
    save_csv(pd.DataFrame(diagnostics), output / "diagnostics.csv")
    save_csv(pd.DataFrame(stability), output / "neighbour_stability.csv")
    save_json({
        "method": "exact", "random_state": 42, "init": "pca", "learning_rate": "auto", "max_iter": 2000,
        "input": "Unstandardised ten-dimensional signed consensus grouped SHAP",
        "primary_all_rows_perplexity": 30, "primary_Test_perplexity": 10,
        "ALL_ROWS_DESCRIPTIVE": "Training-in-sample and Test-out-of-fit; not OOF SHAP",
        "external_cases_embedded": False, "predictive_fits": 0,
        "caption_diagnostic": "Ten-neighbour trustworthiness at all-row perplexity 30",
    }, output / "settings.json")
    check_inputs()
    check_references()
    return pd.DataFrame(diagnostics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shap-dir", type=Path, default=ROOT / "outputs/shap")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tsne")
    args = parser.parse_args()
    run(args.shap_dir, args.output)
    print("Twelve t-SNE embeddings reproduce the accepted coordinates exactly.")
