"""Compare replayed numerical tables with the completed analysis."""

import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from .common import (
    DATA,
    ROOT,
    joblib,
    sha,
    training,
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


def verify_shap_additivity(values, recorded):
    """Check residuals against the contributions, including every descriptor."""
    keys = ["target", "candidate_id", "model_realisation"]
    columns = ["expected_value_pg_ml", "prediction_pg_ml", "grouped_SHAP_pg_ml"]
    if values.empty or not np.isfinite(values[columns].to_numpy(dtype=float)).all():
        raise ValueError("SHAP contributions and predictions must be finite and non-empty")
    rows = []
    for identity, group in values.groupby(keys):
        if len(group) != len(FEATURES) or set(group.descriptor_name) != set(FEATURES):
            raise ValueError(f"Incomplete SHAP descriptors: {identity}")
        for column in columns[:2]:
            np.testing.assert_array_equal(group[column], np.repeat(group[column].iloc[0], len(group)))
        residual = float(group.expected_value_pg_ml.iloc[0] + group.grouped_SHAP_pg_ml.sum() - group.prediction_pg_ml.iloc[0])
        np.testing.assert_allclose(residual, 0, rtol=0, atol=SHAP_TOLERANCE,
                                   err_msg=f"SHAP additivity failed: {identity}")
        rows.append({**dict(zip(keys, identity)), "additivity_residual_pg_ml": residual})
    residuals = pd.DataFrame(rows)
    compare_tables(recorded[residuals.columns], residuals, keys, SHAP_TOLERANCE)
    return float(residuals.additivity_residual_pg_ml.abs().max())


def verify_profile_counts(summary, profiles):
    """Require summary counts to match the checked profile and objective rows."""
    counts = {
        "final_profiles": len(profiles),
        "final_locations": len(profiles[["tnf_mean_pg_ml", "il10_mean_pg_ml"]].drop_duplicates()),
    }
    for name, count in counts.items():
        if summary.get(name) != count:
            raise ValueError(f"Pareto summary {name} differs from checked tables: {summary.get(name)} != {count}")
    return counts


def verify_predictions(predictions, output):
    """Replay saved models and independently reconstruct nearest-neighbor AD."""
    from sklearn.metrics import pairwise_distances
    settings = {row["target"]: row for row in read(output / "inputs/APPLICABILITY_DOMAIN_SETTINGS.json")}
    maximum = 0.0
    for target in SEEDS:
        item = settings[target]
        assert sha(ROOT / item["path"]) == item["sha256"]
        prep = joblib.load(ROOT / item["path"])
        train = prep.transform(training(target)[FEATURES])
        primary = models(target)
        for start in range(0, len(predictions), 16384):
            values = predictions.iloc[start:start + 16384]
            distances = pairwise_distances(prep.transform(values[FEATURES]), train)
            # Partition finds the same nearest five without the inference sort.
            mean5 = np.partition(distances, 4, axis=1)[:, :5].mean(axis=1)
            np.testing.assert_allclose(values[target + "_AD_mean5_distance"], mean5,
                                       rtol=0, atol=ABSOLUTE_TOLERANCE)
            np.testing.assert_allclose(values[target + "_AD_ratio"], mean5 / item["threshold_q95"],
                                       rtol=0, atol=ABSOLUTE_TOLERANCE)
            for seed, pipe in primary:
                actual = pipe.predict(values[FEATURES])
                saved = values[f"{target}_seed_{seed}_pg_ml"].to_numpy()
                np.testing.assert_allclose(actual, saved, rtol=0, atol=ABSOLUTE_TOLERANCE)
                maximum = max(maximum, float(np.max(np.abs(actual - saved))))
        columns = [f"{target}_seed_{seed}_pg_ml" for seed in SEEDS[target]]
        np.testing.assert_array_equal(predictions[target + "_mean_pg_ml"], predictions[columns].mean(axis=1))
        np.testing.assert_array_equal(predictions[target + "_sample_SD_pg_ml"], predictions[columns].std(axis=1, ddof=1))
    return maximum


def verify_representatives(profiles, representatives):
    """Reconstruct endpoint and half-normalized-arc selection from checked rows."""
    xy = ["tnf_mean_pg_ml", "il10_mean_pg_ml"]
    locations = profiles.sort_values(xy + ["candidate_id"]).drop_duplicates(xy)
    if len(locations) < 3:
        if len(representatives):
            raise ValueError("Representatives cannot be invented for fewer than three locations")
        return
    values = locations[xy].to_numpy()
    scaled = (values - values.min(axis=0)) / np.ptp(values, axis=0)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(scaled, axis=0), axis=1))]
    interior = locations.iloc[1:-1].copy()
    interior["distance"] = np.abs(arc[1:-1] - arc[-1] / 2.)
    first = profiles.sort_values(xy + ["candidate_id"], ascending=[True, False, True]).iloc[0]
    middle = interior.sort_values(["distance", "candidate_id"]).iloc[0]
    last = profiles.sort_values([xy[1], xy[0], "candidate_id"], ascending=[False, True, True]).iloc[0]
    assert representatives.display_label.tolist() == ["P1", "P2", "P3"]
    assert representatives.candidate_id.tolist() == [first.candidate_id, middle.candidate_id, last.candidate_id]
    compare_tables(representatives[profiles.columns], profiles.set_index("candidate_id", drop=False)
                   .loc[representatives.candidate_id], ["candidate_id"])


def run(output, reference=None):
    """Verify the current release; compare a matching saved reference if supplied."""
    output = Path(output)
    if reference is not None:
        reference = Path(reference)
    inputs = csv(output / "search/CANDIDATE_INPUTS.csv")
    predictions = csv(output / "results/ALL_CANDIDATES_AND_PREDICTIONS.csv")
    masks = csv(output / "results/COMPUTATIONAL_FRONT_MASKS.csv")
    configuration = read(output / "search/SEARCH_CONFIGURATION.json")
    provenance = read(output / "checks/MODEL_AND_DATA_VERIFICATION.json")
    for name, expected in provenance["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Accepted source changed during Pareto replay: {name}")
    assert configuration["accepted_seeds"] == SEEDS
    assert configuration["AD_settings_sha256"] == sha(output / "inputs/APPLICABILITY_DOMAIN_SETTINGS.json")
    levels = csv(output / "search/CATEGORICAL_LEVEL_SUPPORT.csv")
    allowed = {f: list(q.value) for f, q in levels.loc[levels.included].groupby("descriptor")}
    validate_profiles(inputs, configuration, allowed)
    if not inputs[FEATURES].nunique().gt(1).all():
        raise ValueError("All ten descriptors must vary before inference")
    assert len(inputs) == configuration["declared_finite_grid_size"]
    pd.testing.assert_frame_equal(inputs, predictions[inputs.columns])
    np.testing.assert_array_equal(masks.candidate_id, predictions.candidate_id)
    differences = {"saved_model_prediction_replay": verify_predictions(predictions, output)}
    columns = [f"{t}_seed_{seed}_pg_ml" for t in SEEDS for seed in SEEDS[t]]
    eligible = (predictions.tnf_AD_ratio.le(1) & predictions.il10_AD_ratio.le(1)
                & predictions[columns].ge(0).all(axis=1))
    np.testing.assert_array_equal(eligible, predictions.eligible)
    np.testing.assert_array_equal(eligible, masks.eligible)
    universe = predictions.loc[eligible]
    mean_front = front(universe.tnf_mean_pg_ml, universe.il10_mean_pg_ml)
    np.testing.assert_array_equal(mean_front, masks.loc[eligible, "mean_front"])
    frequency = np.zeros(len(universe), dtype=int)
    for ts, ils in itertools.product(SEEDS["tnf"], SEEDS["il10"]):
        expected = front(universe[f"tnf_seed_{ts}_pg_ml"], universe[f"il10_seed_{ils}_pg_ml"])
        np.testing.assert_array_equal(expected, masks.loc[eligible, f"front_tnf_{ts}_il10_{ils}"])
        frequency += expected
    np.testing.assert_array_equal(frequency, masks.loc[eligible, "nondomination_frequency_of_9"])
    np.testing.assert_array_equal(mean_front & (frequency >= 6), masks.loc[eligible, "final_front"])
    assert not masks.loc[~eligible, ["mean_front", "final_front"]].to_numpy().any()
    checked_tables = {}
    table_specs = [
        ("FINAL_PARETO_PROFILES.csv", ["candidate_id"], ABSOLUTE_TOLERANCE),
        ("FINAL_PARETO_REPRESENTATIVES.csv", ["display_label", "candidate_id"], ABSOLUTE_TOLERANCE),
        ("FINAL_REPRESENTATIVE_SHAP.csv", ["target", "candidate_id", "descriptor_name"], SHAP_TOLERANCE),
        ("REPRESENTATIVE_SHAP_ALL_REALISATIONS.csv", ["target", "candidate_id", "model_realisation", "descriptor_name"], SHAP_TOLERANCE),
        ("REPRESENTATIVE_SHAP_TRANSFORMED.csv", ["target", "candidate_id", "model_realisation", "transformed_feature"], SHAP_TOLERANCE),
    ]
    for filename, keys, tolerance in table_specs:
        checked_tables[filename] = csv(output / "results" / filename)
        if reference is not None:
            differences[filename] = compare_tables(checked_tables[filename], csv(reference / filename), keys, tolerance)
    if reference is not None:
        for filename, values in [("ALL_CANDIDATES_AND_PREDICTIONS.csv", predictions), ("COMPUTATIONAL_FRONT_MASKS.csv", masks)]:
            path = reference / filename
            if not path.exists():
                path = path.with_suffix(path.suffix + ".gz")
            differences[filename] = compare_tables(values, csv(path), ["candidate_id"])
    profiles = checked_tables["FINAL_PARETO_PROFILES.csv"]
    compare_tables(profiles[predictions.columns], predictions.loc[masks.final_front], ["candidate_id"])
    expected_frequency = masks.set_index("candidate_id").loc[profiles.candidate_id, "nondomination_frequency_of_9"]
    np.testing.assert_array_equal(profiles.nondomination_frequency_of_9, expected_frequency)
    verify_representatives(profiles, checked_tables["FINAL_PARETO_REPRESENTATIVES.csv"])
    residuals = csv(output / "checks/SHAP_ADDITIVITY.csv")
    shap_values = checked_tables["REPRESENTATIVE_SHAP_ALL_REALISATIONS.csv"]
    maximum_residual = verify_shap_additivity(shap_values, residuals)
    for target in SEEDS:
        actual = set(shap_values.loc[shap_values.target.eq(target), "model_realisation"].astype(str))
        assert actual == {str(seed) for seed in SEEDS[target]} | {"mean"}
    summary = read(output / "results/PARETO_SUMMARY.json")
    counts = verify_profile_counts(summary, profiles)
    report = {
        "generated": len(inputs), "eligible": int(eligible.sum()),
        "retained_profiles": counts["final_profiles"], "objective_locations": counts["final_locations"],
        "accepted_seeds": SEEDS, "identity_sets_exact": True,
        "memberships_and_frequencies_exact": True, "all_ten_descriptors_vary": True,
        "current_model_predictions_replayed": True,
        "reference_comparison": str(reference) if reference is not None else None,
        "max_absolute_differences": differences, "max_SHAP_additivity_residual_pg_ml": maximum_residual,
        "numeric_absolute_tolerance": ABSOLUTE_TOLERANCE, "SHAP_absolute_tolerance": SHAP_TOLERANCE,
        "relative_tolerance": 0, "model_fits": 0, "preprocessing_fits": 0,
        "figures_generated": 0,
    }
    dump(output / "checks/NUMERICAL_REPRODUCTION.json", report)
    print(report, flush=True)
