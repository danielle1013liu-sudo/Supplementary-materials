"""Evaluation boundaries and aggregation rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydrogel import cases, evaluation
from hydrogel.data import ROOT, load_dataset, load_p32_scope


def test_fold_preprocessing_and_joint_prediction_batch(monkeypatch):
    frame = pd.DataFrame({
        "record_id": [f"r{i}" for i in range(10)],
        "x": [np.nan, *range(1, 10)],
        "cytokine_pg_ml": np.arange(10, 20, dtype=float),
    })
    train_ids = ["r4", "r0", "r6", "r1", "r7", "r2", "r5", "r3"]
    labels = np.array([1, 2] * 4)
    specification = {
        "target": "tnf", "model_seeds": [101, 202, 303], "n_splits": 2,
        "models": {"RF": {"imputer": "median", "parameters": {}}},
        "imputation_methods": ["median"],
    }
    case_frame = pd.DataFrame({
        "record_id": ["C4_Group2", "C4_Group4"],
        "x": [100.0, 200.0], "observed_pg_ml": [5.0, 7.0],
    })
    monkeypatch.setattr(evaluation, "load_config", lambda *args, **kwargs: specification)
    monkeypatch.setattr(evaluation, "load_schema", lambda **kwargs: {"predictors": ["x"]})
    monkeypatch.setattr(evaluation, "load_dataset", lambda *args, **kwargs: frame)
    monkeypatch.setattr(evaluation, "load_split", lambda *args, **kwargs: {
        "train_ids": train_ids, "test_ids": ["r9", "r8"], "fold_labels": labels.tolist(),
    })
    monkeypatch.setattr(evaluation, "load_c4", lambda *args, **kwargs: case_frame)
    imputer_fits, estimator_fits, prediction_batches = [], [], []

    class RecordingImputer:
        def fit_transform(self, values, outcome):
            imputer_fits.append(values.index.tolist())
            self.median = float(values.x.median())
            return self.transform(values)

        def transform(self, values):
            return values.fillna(self.median).to_numpy(dtype=float)

    class RecordingEstimator:
        def __init__(self, seed):
            self.seed = seed

        def fit(self, values, outcome):
            estimator_fits.append((self.seed, len(outcome)))
            self.mean = float(np.mean(outcome))
            return self

        def predict(self, values):
            prediction_batches.append((self.seed, len(values)))
            return np.full(len(values), self.mean + self.seed / 1000.0)

    monkeypatch.setattr(evaluation, "make_preprocessor", lambda *args: RecordingImputer())
    monkeypatch.setattr(evaluation, "make_estimator", lambda family, parameters, seed: RecordingEstimator(seed))
    result = evaluation.evaluate_configuration("tnf", "RF")

    assert imputer_fits == [train_ids, train_ids[1::2], train_ids[::2]]
    assert estimator_fits == [(seed, n) for seed in [101, 202, 303] for n in [8, 4, 4]]
    assert prediction_batches == [(seed, n) for seed in [101, 202, 303] for n in [12, 4, 4]]
    assert len(result.metrics) == 12
    assert len(result.predictions) == 60
    assert list(result.models) == [101, 202, 303]
    y_train = frame.set_index("record_id").loc[train_ids, "cytokine_pg_ml"].to_numpy()
    for seed in specification["model_seeds"]:
        predicted = np.array([y_train[labels != fold].mean() + seed / 1000.0 for fold in labels])
        rows = result.predictions.query("model_seed == @seed and scope == 'Validation'")
        assert rows.record_id.tolist() == train_ids
        np.testing.assert_array_equal(rows.predicted_pg_ml, predicted)
        row = result.metrics.query("model_seed == @seed and scope == 'Validation'").iloc[0]
        expected = evaluation.regression_metrics(y_train, predicted)
        for metric, value in expected.items():
            assert row[metric] == value


def test_partition_rejects_overlapping_memberships():
    frame = pd.DataFrame({"record_id": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="overlap"):
        evaluation._partition(frame, {"train_ids": ["a", "b"], "test_ids": ["b", "c"], "fold_labels": [1, 2]}, 2)


def test_summary_uses_sample_sd_of_individual_metrics():
    records = [
        {"target": "tnf", "family": "RF", "imputer": "median", "scope": "Test",
         "model_seed": seed, "R2": value, "RMSE_pg_ml": value + 10, "MAE_pg_ml": value + 5}
        for seed, value in zip([11, 22, 33], [0.2, 0.5, 0.8], strict=True)
    ]
    result = evaluation.summarize_metrics(pd.DataFrame(records)).iloc[0]
    assert result.n_models == 3
    assert result.R2_mean == pytest.approx(0.5)
    assert result.R2_SD == pytest.approx(0.3)
    with pytest.raises(ValueError, match="three individual"):
        evaluation.summarize_metrics(pd.DataFrame(records[:2]))


def test_p32_scope_rejects_a_modified_membership_file(tmp_path):
    directory = tmp_path / "data" / "splits"
    directory.mkdir(parents=True)
    (tmp_path / "data" / "manifest.json").write_bytes((ROOT / "data" / "manifest.json").read_bytes())
    destination = directory / "tnf_p32.json"
    destination.write_bytes((ROOT / "data" / "splits" / "tnf_p32.json").read_bytes())
    assert len(load_p32_scope("tnf", root=tmp_path)["fit_ids"]) == 148
    destination.write_bytes(destination.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        load_p32_scope("tnf", root=tmp_path)


@pytest.mark.parametrize("target,expected_count", [("tnf", 148), ("il10", 129)])
def test_p32_uses_saved_whole_study_population(monkeypatch, target, expected_count):
    frame = load_dataset(target)
    expected_ids = frame.loc[~frame.source_doi.eq(cases.P32_DOI), "record_id"].tolist()
    calls = []

    class RecordingPipeline:
        def fit(self, features, response):
            assert len(features) == expected_count
            assert features.index.tolist() == expected_ids
            calls.append(("fit", len(features)))
            return self

        def predict(self, features):
            assert features.index.tolist() == cases.P32_CASE_IDS[target]
            assert pd.isna(features.iloc[0].Gprime_kPa)
            calls.append(("predict", len(features)))
            return np.array([200.0, 300.0])

    monkeypatch.setattr(cases, "make_pipeline", lambda *args: RecordingPipeline())
    result = cases.evaluate_p32(target)
    assert calls == [(operation, count) for _ in range(3) for operation, count in [("fit", expected_count), ("predict", 2)]]
    assert set(result.metrics.scope) == {"P32"}
    assert len(result.metrics) == 3
    assert len(result.predictions) == 6
