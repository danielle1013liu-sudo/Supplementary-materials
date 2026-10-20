"""Check primary Test/C4 and separate P15 saved-model predictions without fitting."""

import argparse
import pickle
import zlib
from pathlib import Path
import numpy as np
from .data import check_inputs, configuration, split_data, load_cases, load_data
from .models import make_pipeline
from .pareto.common import (
    ROOT,
    DATA,
    FEATURES,
    KEY,
    SEEDS,
    csv,
    dataset,
    describe,
    dump,
    models,
    read,
    sha,
    table,
)


def run(output):
    output = Path(output)
    check_inputs()
    checks = []
    with np.load(ROOT / "data/reference/accepted_predictions.npz") as reference:
        for target in SEEDS:
            _, test = split_data(target)
            cases = load_cases(target)
            for seed, model in models(target):
                if describe(model) != describe(make_pipeline(target, seed)):
                    raise ValueError(f"Training configuration differs: {target}, {seed}")
                for scope, frame in [("Test", test[FEATURES]), ("C4", cases[FEATURES])]:
                    expected = reference[f"{target}_{seed}_{scope}"]
                    prediction = model.predict(frame)
                    np.testing.assert_array_equal(prediction, expected)
                    checks.append({
                        "target": target, "realisation": seed, "scope": scope,
                        "n": len(frame),
                        "max_absolute_difference_pg_ml": float(np.max(np.abs(prediction - expected))),
                    })
    p15 = ROOT / "data/p15"
    for path, expected_hash in read(p15 / "checksums.json").items():
        if sha(ROOT / path) != expected_hash:
            raise ValueError(f"Changed P15 input: {path}")
    manifest = read(p15 / "study_exclusion.json")
    reference = csv(p15 / "study_holdout_predictions.csv")
    for target, settings in manifest["targets"].items():
        values = load_data(target).set_index(KEY, drop=False)
        if set(settings["fit_IDs_ordered"]) & set(settings["excluded_IDs"]):
            raise ValueError("P15 source rows occur in the fitting membership")
        if set(values.index) - set(settings["excluded_IDs"]) != set(
            settings["fit_IDs_ordered"]
        ):
            raise ValueError(
                "P15 fitting membership differs from whole-study exclusion"
            )
        cases = values.loc[settings["pair_IDs"], FEATURES]
        if set(cases.stimulation_context) != {"unstimulated_or_basal"}:
            raise ValueError("P15 must use the basal condition")
        predictions = []
        for seed in settings["fixed_estimator_seeds"]:
            path = ROOT / f"models/p15/{target}_{seed}_P15_excluded.pklz"
            model = pickle.loads(zlib.decompress(path.read_bytes()))
            prediction = model.predict(cases)
            expected = (
                reference.loc[
                    reference.target.eq(target)
                    & reference.model_realisation.astype(str).eq(str(seed))
                ]
                .set_index(KEY)
                .loc[settings["pair_IDs"], "predicted_pg_ml"]
                .to_numpy()
            )
            np.testing.assert_allclose(prediction, expected, atol=1e-10, rtol=0)
            predictions.append(prediction)
            checks.append(
                {
                    "target": target,
                    "realisation": seed,
                    "scope": "P15 whole-study exclusion",
                    "n": len(cases),
                    "max_absolute_difference_pg_ml": float(
                        np.max(np.abs(prediction - expected))
                    ),
                }
            )
        expected = (
            reference.loc[
                reference.target.eq(target)
                & reference.model_realisation.astype(str).eq("ensemble_mean")
            ]
            .set_index(KEY)
            .loc[settings["pair_IDs"]]
        )
        np.testing.assert_allclose(
            np.mean(predictions, axis=0), expected.predicted_pg_ml, atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(
            np.std(predictions, axis=0, ddof=1),
            expected.prediction_sample_SD_pg_ml,
            atol=1e-10,
            rtol=0,
        )
    table(output / "CASE_AND_TEST_REPLAY.csv", checks)
    dump(
        output / "CASE_AND_TEST_REPLAY.json",
        {
            "checks": checks,
            "predictive_fits": 0,
            "preprocessing_fits": 0,
            "primary_Test_and_C4_same_objects": True,
            "P15_separate_study_excluded_models": True,
        },
    )
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.output)
