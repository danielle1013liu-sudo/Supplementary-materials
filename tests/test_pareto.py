import itertools
import numpy as np
import pandas as pd
import pytest
from src.pareto.common import DATA, FEATURES, NUM, SEEDS, csv, digest, front, read
from src.pareto.verification import compare_tables, validate_profiles, verify_representatives
from src.data import configuration
from src.case_replay import run as replay_cases
from src.hydrogel_immune_ml.public_encoding import (
    PUBLIC_VERSION,
    VERSION_COLUMN,
    decode_public_input,
    encode_public_input,
)


@pytest.mark.parametrize(
    "x,y,expected",
    [
        ([1, 2, 3], [3, 2, 1], [True, False, False]),
        ([1, 2, 3], [1, 2, 3], [True, True, True]),
        ([1, 1, 1, 2], [2, 2, 1, 2], [True, True, False, False]),
        ([0, 0, 1, 1], [0, 1, 1, 2], [False, True, False, True]),
        ([], [], []),
    ],
)
def test_dominance_and_exact_ties(x, y, expected):
    np.testing.assert_array_equal(front(x, y), expected)


def test_global_front_is_not_concatenated_group_fronts():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([4.0, 5.0, 2.0, 3.0])
    subgroup = np.r_[front(x[:2], y[:2]), front(x[2:], y[2:])]
    assert subgroup.all()
    np.testing.assert_array_equal(front(x, y), [True, True, False, False])


def test_global_batch_reduction_preserves_ties():
    rng = np.random.default_rng(4)
    x = rng.integers(0, 12, 120)
    y = rng.integers(0, 12, 120)
    local = np.concatenate(
        [
            i + np.flatnonzero(front(x[i : i + 17], y[i : i + 17]))
            for i in range(0, len(x), 17)
        ]
    )
    merged = np.zeros(len(x), bool)
    merged[local[front(x[local], y[local])]] = True
    np.testing.assert_array_equal(merged, front(x, y))


def test_identity_and_cache_depend_on_every_descriptor():
    config = read(DATA / "reference/SEARCH_CONFIGURATION.json")
    row = csv(DATA / "reference/FINAL_PARETO_REPRESENTATIVES.csv").iloc[0]
    values = [float(row[f]) if f in NUM else row[f] for f in FEATURES]
    original = digest({"schema_version": config["schema_version"], "profile": values})
    assert row.candidate_id == "A10_" + original[:20]
    cache = digest(
        {
            "configuration": config["cache_key"],
            "batch_number": 0,
            "all_profiles": [original],
        }
    )
    for index, field in enumerate(FEATURES):
        altered = values.copy()
        altered[index] = (
            altered[index] + 1 if field in NUM else altered[index] + "_different"
        )
        identity = digest(
            {"schema_version": config["schema_version"], "profile": altered}
        )
        assert identity != original
        assert (
            digest(
                {
                    "configuration": config["cache_key"],
                    "batch_number": 0,
                    "all_profiles": [identity],
                }
            )
            != cache
        )


def test_unsupported_category_is_rejected():
    config = read(DATA / "reference/SEARCH_CONFIGURATION.json")
    row = csv(DATA / "reference/FINAL_PARETO_REPRESENTATIVES.csv").iloc[:1].copy()
    allowed = {"macrophage_model_class": set(row.macrophage_model_class)}
    row.loc[row.index[0], "macrophage_model_class"] = "unsupported_cell_line"
    with pytest.raises(ValueError, match="Unsupported categories"):
        validate_profiles(row, config, allowed)


def test_profile_id_cannot_hide_a_changed_feature():
    config = read(DATA / "reference/SEARCH_CONFIGURATION.json")
    row = csv(DATA / "reference/FINAL_PARETO_REPRESENTATIVES.csv").iloc[:1].copy()
    row.loc[row.index[0], "culture_or_exposure_duration_h"] += 1
    with pytest.raises(ValueError, match="identity does not match"):
        validate_profiles(row, config, {})


def test_accepted_model_panel_is_read_from_canonical_configuration():
    assert SEEDS == {target: configuration(target)["seeds"] for target in ("tnf", "il10")}
    assert all(len(set(values)) == 3 for values in SEEDS.values())


def test_current_profiles_round_trip_publication_codes(replay_dir):
    profiles = csv(replay_dir / "results/FINAL_PARETO_PROFILES.csv")
    representatives = csv(replay_dir / "results/FINAL_PARETO_REPRESENTATIVES.csv")
    for frame in (profiles, representatives):
        coded = encode_public_input(frame)
        assert coded[VERSION_COLUMN].eq(PUBLIC_VERSION).all()
        pd.testing.assert_frame_equal(decode_public_input(coded, PUBLIC_VERSION), frame,
                                      check_exact=True)
    verify_representatives(profiles, representatives)


def test_all_ten_inputs_and_full_candidate_identity_order(replay_dir):
    inputs = csv(replay_dir / "search/CANDIDATE_INPUTS.csv")
    reference = csv(replay_dir / "results/ALL_CANDIDATES_AND_PREDICTIONS.csv")
    assert inputs.columns[1:11].tolist() == FEATURES
    assert inputs[FEATURES].nunique().gt(1).all()
    pd.testing.assert_frame_equal(inputs, reference[inputs.columns])


def test_all_six_predictions_control_eligibility(replay_dir):
    values = csv(replay_dir / "results/ALL_CANDIDATES_AND_PREDICTIONS.csv")
    columns = [f"{t}_seed_{s}_pg_ml" for t in SEEDS for s in SEEDS[t]]
    expected = (
        values[columns].ge(0).all(axis=1)
        & values.tnf_AD_ratio.le(1)
        & values.il10_AD_ratio.le(1)
    )
    np.testing.assert_array_equal(values.eligible, expected)
    eligible = values.loc[values.eligible].iloc[:1].copy()
    for column in columns:
        altered = eligible.copy()
        altered[column] = -1e-12
        assert not altered[columns].ge(0).all(axis=1).iloc[0]


def test_every_pair_uses_complete_eligible_universe(replay_dir):
    values = csv(replay_dir / "results/ALL_CANDIDATES_AND_PREDICTIONS.csv")
    masks = csv(replay_dir / "results/COMPUTATIONAL_FRONT_MASKS.csv")
    eligible = values.loc[values.eligible]
    frequency = np.zeros(len(eligible), int)
    for t, i in itertools.product(SEEDS["tnf"], SEEDS["il10"]):
        expected = front(
            eligible[f"tnf_seed_{t}_pg_ml"], eligible[f"il10_seed_{i}_pg_ml"]
        )
        np.testing.assert_array_equal(
            expected, masks.loc[eligible.index, f"front_tnf_{t}_il10_{i}"]
        )
        frequency += expected
    np.testing.assert_array_equal(
        frequency, masks.loc[eligible.index, "nondomination_frequency_of_9"]
    )
    expected = front(eligible.tnf_mean_pg_ml, eligible.il10_mean_pg_ml) & (
        frequency >= 6
    )
    np.testing.assert_array_equal(expected, masks.loc[eligible.index, "final_front"])


def test_retained_ids_frequencies_and_representatives(replay_dir):
    profiles = csv(replay_dir / "results/FINAL_PARETO_PROFILES.csv")
    reps = csv(replay_dir / "results/FINAL_PARETO_REPRESENTATIVES.csv")
    masks = csv(replay_dir / "results/COMPUTATIONAL_FRONT_MASKS.csv")
    assert set(profiles.candidate_id) == set(masks.loc[masks.final_front, "candidate_id"])
    expected = masks.set_index("candidate_id").loc[profiles.candidate_id, "nondomination_frequency_of_9"]
    np.testing.assert_array_equal(profiles.nondomination_frequency_of_9, expected)
    verify_representatives(profiles, reps)
    report = read(replay_dir / "checks/NUMERICAL_REPRODUCTION.json")
    assert report["retained_profiles"] == len(profiles)
    assert report["objective_locations"] == len(profiles[["tnf_mean_pg_ml", "il10_mean_pg_ml"]].drop_duplicates())
    assert report["accepted_seeds"] == SEEDS


def test_current_ad_is_revalidated_without_predictor_or_diagnostic_fits(replay_dir):
    report = read(replay_dir / "checks/MODEL_AND_DATA_VERIFICATION.json")
    assert report["predictive_fits"] == report["diagnostic_fits"] == 0
    assert all(row["learned_statistics_revalidated"] for row in report["applicability_domain"])
    assert all(row["Test_predictions_exact"] for row in report["models"])
    config = read(replay_dir / "search/SEARCH_CONFIGURATION.json")
    assert config["accepted_seeds"] == SEEDS
    assert config["AD_settings_sha256"]


def test_grouped_shap_matches_sources_and_is_additive(replay_dir):
    values = csv(replay_dir / "results/REPRESENTATIVE_SHAP_ALL_REALISATIONS.csv")
    for target in SEEDS:
        actual = set(values.loc[values.target.eq(target), "model_realisation"].astype(str))
        assert actual == {str(seed) for seed in SEEDS[target]} | {"mean"}
    for _, group in values.groupby(["target", "candidate_id", "model_realisation"]):
        assert set(group.descriptor_name) == set(FEATURES)
        assert (
            abs(
                group.expected_value_pg_ml.iloc[0]
                + group.grouped_SHAP_pg_ml.sum()
                - group.prediction_pg_ml.iloc[0]
            )
            < 1e-7
        )


def test_primary_test_c4_and_separate_p15_replay(tmp_path):
    checks = replay_cases(tmp_path)
    assert len(checks) == 18
    assert max(c["max_absolute_difference_pg_ml"] for c in checks) <= 1e-10
