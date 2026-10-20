"""Regression checks for published explanations and case/model separation."""

import os

import numpy as np
import pandas as pd
import pytest

from src.data import ROOT, ID, feature_names, load_data, read_csv, split_definition
from src.interpretation import REFERENCE, check_references
from src.shap_analysis import run as export_shap
from src.case_studies import run as export_cases
from src.tsne_diagnostics import caption_trustworthiness


def test_primary_shap_reproduces_frozen_arrays_without_fitting(tmp_path):
    # The suite-wide fit guard also catches accidental preprocessing refits.
    result = pd.DataFrame(export_shap(tmp_path))
    assert result.groupby("target").n_Test.first().to_dict() == {"tnf": 28, "il10": 27}
    np.testing.assert_allclose(result.groupby("target").importance_percent.sum(), 100, atol=1e-12, rtol=0)
    for target in ("tnf", "il10"):
        all_rows = read_csv(tmp_path / "ALL_ROWS_DESCRIPTIVE" / f"{target}_consensus.csv")
        test = read_csv(tmp_path / "TEST_ONLY" / f"{target}_consensus.csv")
        assert set(test[ID]) == set(split_definition(target)["Test_IDs"])
        np.testing.assert_array_equal(test[ID], split_definition(target)["Test_IDs"])
        assert len(all_rows) == len(load_data(target))
        np.testing.assert_allclose(
            all_rows.expected_value_mean_pg_ml + all_rows[feature_names()].sum(axis=1),
            all_rows.prediction_mean_pg_ml, atol=1e-7, rtol=0,
        )


def test_case_studies_use_the_two_distinct_model_populations(tmp_path):
    summary = export_cases(tmp_path)
    assert len(summary) == 8
    p15 = summary[summary.study.eq("P15")]
    c4 = summary[summary.study.eq("C4")]
    assert set(p15.membership) == {"whole-study-excluded"}
    assert set(c4.membership) == {"external"}
    assert set(p15[ID]) == {"TNF-0035", "TNF-0036", "IL10-0029", "IL10-0030"}
    assert set(p15.formulation) == {"PEG microgel scaffold", "PEG microgel scaffold with tethered IL-10"}
    # Table S7b, at its printed precision. This fails if primary P15 models or
    # the superseded IL-4/IL-13 case are substituted for whole-study exclusion.
    np.testing.assert_allclose(
        p15.prediction_mean_pg_ml, [422.555, 190.882, 624.815, 716.929], atol=0.0005, rtol=0,
    )
    np.testing.assert_allclose(
        c4.prediction_mean_pg_ml, [590.973, 487.348, 167.050, 183.802], atol=0.0005, rtol=0,
    )
    delta = read_csv(tmp_path / "delta_shap.csv")
    for _, group in delta.groupby(["study", "target"]):
        np.testing.assert_allclose(group.mean_delta_SHAP_pg_ml.sum(), group.predicted_change_mean_pg_ml.iloc[0], atol=1e-7, rtol=0)
        assert group.raw_descriptor_changed.sum() == 1


@pytest.mark.parametrize("target,n,expected", [("tnf", 139, 0.9236769289022223), ("il10", 133, 0.9801951687729963)])
def test_main_tsne_coordinates_and_caption_diagnostic(target, n, expected):
    check_references()
    frame = read_csv(ROOT / f"outputs/tsne/ALL_ROWS_DESCRIPTIVE/{target}_p30.csv")
    reference = read_csv(REFERENCE / f"{target}_ALL_ROWS_DESCRIPTIVE_p30.csv")
    assert len(frame) == n and frame[ID].is_unique
    np.testing.assert_array_equal(frame[["tsne_1", "tsne_2"]], reference[["tsne_1", "tsne_2"]])
    pd.testing.assert_frame_equal(frame[feature_names()], load_data(target)[feature_names()])
    original = read_csv(ROOT / f"outputs/shap/ALL_ROWS_DESCRIPTIVE/{target}_consensus.csv")[feature_names()].to_numpy()
    embedding = frame[["tsne_1", "tsne_2"]].to_numpy()
    trustworthiness = caption_trustworthiness(original, embedding)
    np.testing.assert_allclose(trustworthiness, expected, atol=1e-15, rtol=0)


def test_caption_diagnostic_keeps_inputs_and_cpu_settings(monkeypatch):
    monkeypatch.setenv("NPY_DISABLE_CPU_FEATURES", "FMA3")
    original = np.arange(30, dtype=float)[:, None]
    embedding = np.column_stack((original, np.zeros(30)))
    original_before, embedding_before = original.copy(), embedding.copy()
    assert caption_trustworthiness(original, embedding) == 1
    assert os.environ["NPY_DISABLE_CPU_FEATURES"] == "FMA3"
    np.testing.assert_array_equal(original, original_before)
    np.testing.assert_array_equal(embedding, embedding_before)


def test_tsne_sensitivity_covers_all_prespecified_settings():
    diagnostics = read_csv(ROOT / "outputs/tsne/diagnostics.csv")
    expected = {(target, scope, p) for target in ("tnf", "il10")
                for scope, ps in [("TEST_ONLY", (5, 10, 15)), ("ALL_ROWS_DESCRIPTIVE", (10, 20, 30))] for p in ps}
    assert set(zip(diagnostics.target, diagnostics.scope, diagnostics.perplexity)) == expected
    assert diagnostics.exact_coordinate_reproduction.all()
    assert diagnostics.KL_divergence.ge(0).all()
    assert diagnostics.actual_iterations.le(2000).all()
