"""Compare replayed numerical tables with the completed analysis."""

import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from .common import (
    DATA,
    FEATURES,
    NUM,
    SEEDS,
    csv,
    digest,
    dump,
    front,
    models,
    read,
    table,
)

ABSOLUTE_TOLERANCE = 1e-10
SHAP_TOLERANCE = 1e-7


def compare_tables(actual, expected, keys, tolerance=ABSOLUTE_TOLERANCE):
    """Require identical keyed rows and categories; bound numerical differences."""
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError(f"Duplicate row identities: {keys}")
    a = actual.set_index(keys).sort_index()
    b = expected.set_index(keys).sort_index()
    pd.testing.assert_index_equal(a.index, b.index)
    if set(a.columns) != set(b.columns):
        raise ValueError("Source and reproduced table columns differ")
    maximum = 0.0
    for name in b.columns:
        if pd.api.types.is_numeric_dtype(b[name]) and not pd.api.types.is_bool_dtype(
            b[name]
        ):
            np.testing.assert_allclose(
                a[name], b[name], rtol=0, atol=tolerance, equal_nan=True, err_msg=name
            )
            delta = np.abs(
                a[name].to_numpy(dtype=float) - b[name].to_numpy(dtype=float)
            )
            if np.isfinite(delta).any():
                maximum = max(maximum, float(np.nanmax(delta)))
        else:
            pd.testing.assert_series_equal(a[name], b[name], check_dtype=False)
    return maximum


def validate_profiles(values, configuration, allowed):
    """Reject unresolved or unsupported inputs and verify all ten identity fields."""
    if values[FEATURES].isna().any().any():
        raise ValueError("Candidate descriptors must be complete")
    for field, supported in allowed.items():
        unknown = set(values[field]) - set(supported)
        if unknown:
            raise ValueError(f"Unsupported categories for {field}: {sorted(unknown)}")
    for row in values.itertuples(index=False):
        profile = [
            float(getattr(row, f)) if f in NUM else getattr(row, f) for f in FEATURES
        ]
        identity = digest(
            {"schema_version": configuration["schema_version"], "profile": profile}
        )
        if row.profile_sha256 != identity or row.candidate_id != "A10_" + identity[:20]:
            raise ValueError(
                f"Candidate identity does not match its ten inputs: {row.candidate_id}"
            )


def run(output):
    output = Path(output)
    reference = DATA / "reference"
    inputs = csv(output / "search/CANDIDATE_INPUTS.csv")
    predictions = csv(output / "results/ALL_CANDIDATES_AND_PREDICTIONS.csv")
    masks = csv(output / "results/COMPUTATIONAL_FRONT_MASKS.csv")
    configuration = read(output / "search/SEARCH_CONFIGURATION.json")
    levels = csv(output / "search/CATEGORICAL_LEVEL_SUPPORT.csv")
    allowed = {
        f: list(q.value) for f, q in levels.loc[levels.included].groupby("descriptor")
    }
    validate_profiles(inputs, configuration, allowed)
    if not inputs[FEATURES].nunique().gt(1).all():
        raise ValueError("All ten descriptors must vary before inference")
    pd.testing.assert_frame_equal(inputs, predictions[inputs.columns])
    expected = csv(reference / "ALL_CANDIDATES_AND_PREDICTIONS.csv.gz")
    differences = {
        "candidate_predictions_and_AD": compare_tables(
            predictions, expected, ["candidate_id"]
        )
    }
    pd.testing.assert_frame_equal(
        masks, csv(reference / "COMPUTATIONAL_FRONT_MASKS.csv.gz")
    )
    columns = [f"{t}_seed_{s}_pg_ml" for t in SEEDS for s in SEEDS[t]]
    eligible = (
        predictions.tnf_AD_ratio.le(1)
        & predictions.il10_AD_ratio.le(1)
        & predictions[columns].ge(0).all(axis=1)
    )
    np.testing.assert_array_equal(eligible, predictions.eligible)
    universe = predictions.loc[eligible]
    np.testing.assert_array_equal(
        front(universe.tnf_mean_pg_ml, universe.il10_mean_pg_ml),
        masks.loc[eligible, "mean_front"],
    )
    for ts, ils in itertools.product(SEEDS["tnf"], SEEDS["il10"]):
        np.testing.assert_array_equal(
            front(universe[f"tnf_seed_{ts}_pg_ml"], universe[f"il10_seed_{ils}_pg_ml"]),
            masks.loc[eligible, f"front_tnf_{ts}_il10_{ils}"],
        )
    for filename, keys, tolerance in [
        ("FINAL_PARETO_PROFILES.csv", ["candidate_id"], ABSOLUTE_TOLERANCE),
        (
            "FINAL_PARETO_REPRESENTATIVES.csv",
            ["display_label", "candidate_id"],
            ABSOLUTE_TOLERANCE,
        ),
        (
            "FINAL_REPRESENTATIVE_SHAP.csv",
            ["target", "candidate_id", "descriptor_name"],
            SHAP_TOLERANCE,
        ),
        (
            "REPRESENTATIVE_SHAP_ALL_REALISATIONS.csv",
            ["target", "candidate_id", "model_realisation", "descriptor_name"],
            SHAP_TOLERANCE,
        ),
        (
            "REPRESENTATIVE_SHAP_TRANSFORMED.csv",
            ["target", "candidate_id", "model_realisation", "transformed_feature"],
            SHAP_TOLERANCE,
        ),
    ]:
        differences[filename] = compare_tables(
            csv(output / "results" / filename),
            csv(reference / filename),
            keys,
            tolerance,
        )
    residuals = csv(output / "checks/SHAP_ADDITIVITY.csv")
    summary = read(output / "results/PARETO_SUMMARY.json")
    report = {
        "generated": len(inputs),
        "eligible": int(eligible.sum()),
        "retained_profiles": summary["final_profiles"],
        "objective_locations": summary["final_locations"],
        "identity_sets_exact": True,
        "memberships_and_frequencies_exact": True,
        "all_ten_descriptors_vary": True,
        "max_absolute_differences": differences,
        "max_SHAP_additivity_residual_pg_ml": float(
            residuals.additivity_residual_pg_ml.abs().max()
        ),
        "numeric_absolute_tolerance": ABSOLUTE_TOLERANCE,
        "SHAP_absolute_tolerance": SHAP_TOLERANCE,
        "relative_tolerance": 0,
        "model_fits": 0,
        "preprocessing_fits": 0,
        "figures_generated": 0,
    }
    dump(output / "checks/NUMERICAL_REPRODUCTION.json", report)
    print(report, flush=True)
