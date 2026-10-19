import json

import numpy as np
import pandas as pd
import pytest

from src.data import ROOT, configuration
from src.hydrogel_immune_ml import model_comparison
from src.hydrogel_immune_ml.model_comparison import (
    FAMILIES, REFERENCE_TOLERANCE, export_primary_test, export_validation,
    load_evidence, matched_records, primary_test_records, validation_summary,
)


def test_eight_archived_summaries_without_seed_display(tmp_path):
    benchmarks, comparisons = load_evidence()
    output = tmp_path / 'validation.csv'
    summary = export_validation(output)
    assert b'\r\n' not in output.read_bytes()
    assert len(summary) == 8
    assert set(summary.scope) == {'Validation'}
    assert all('seed' not in column for column in summary.columns)
    for target in ('tnf', 'il10'):
        assert set(summary.loc[summary.target.eq(target), 'family']) == set(FAMILIES)
    np.testing.assert_allclose(
        summary[['Validation_R2_mean', 'Validation_R2_sample_SD']],
        benchmarks[['Validation_R2_mean', 'Validation_R2_sample_SD']],
        atol=REFERENCE_TOLERANCE, rtol=0,
    )
    for index, benchmark in benchmarks.iterrows():
        selected = matched_records(benchmark, comparisons)
        for name in ('RMSE', 'MAE'):
            values = selected[f'Validation_{name}'].to_numpy()
            assert summary.loc[index, f'Validation_{name}_mean_pg_ml'] == np.mean(values)
            assert summary.loc[index, f'Validation_{name}_sample_SD_pg_ml'] == np.std(values, ddof=1)
    pd.testing.assert_frame_equal(pd.read_csv(output, float_precision='round_trip'), summary, check_exact=True)


def test_other_imputers_and_configurations_are_not_pooled():
    benchmarks, comparisons = load_evidence()
    expected = validation_summary(benchmarks, comparisons)
    retained = set()
    for _, benchmark in benchmarks.iterrows():
        retained.update(matched_records(benchmark, comparisons).index)
    other = ~comparisons.index.isin(retained)
    comparisons.loc[other, ['Validation_R2', 'Validation_RMSE', 'Validation_MAE']] = -999
    pd.testing.assert_frame_equal(validation_summary(benchmarks, comparisons), expected, check_exact=True)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'wrong_config', 'wrong_identity', 'parameters', 'metric'])
def test_rejects_incomplete_or_changed_comparator_evidence(change):
    benchmarks, comparisons = load_evidence()
    selected = matched_records(benchmarks.iloc[0], comparisons)
    index = selected.index[0]
    if change == 'missing':
        comparisons = comparisons.drop(index)
    elif change == 'duplicate':
        comparisons = pd.concat([comparisons, comparisons.loc[[index]]], ignore_index=True)
    elif change == 'wrong_config':
        comparisons.loc[index, 'config_hash'] = 'wrong_configuration'
    elif change == 'wrong_identity':
        comparisons.loc[index, 'estimator_seed'] = selected.estimator_seed.iloc[1]
    elif change == 'parameters':
        parameters = json.loads(comparisons.loc[index, 'parameters'])
        parameters['max_depth'] += 1
        comparisons.loc[index, 'parameters'] = json.dumps(parameters)
    else:
        comparisons.loc[index, 'Validation_R2'] += 0.01
    with pytest.raises(ValueError):
        validation_summary(benchmarks, comparisons)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'identities'])
def test_rejects_incomplete_or_changed_benchmark(change):
    benchmarks, comparisons = load_evidence()
    if change == 'missing':
        benchmarks = benchmarks.iloc[1:]
    elif change == 'duplicate':
        benchmarks = pd.concat([benchmarks.iloc[1:], benchmarks.iloc[[1]]], ignore_index=True)
    else:
        benchmarks.loc[0, 'shared_seeds'] = '[1, 2, 3]'
    with pytest.raises(ValueError):
        validation_summary(benchmarks, comparisons)


def test_primary_test_export_is_separate_from_validation(tmp_path):
    validation = export_validation(tmp_path / 'validation.csv')
    test = export_primary_test(tmp_path / 'test.csv')
    assert b'\r\n' not in (tmp_path / 'test.csv').read_bytes()
    assert len(test) == 2
    assert set(test.scope) == {'Test'}
    assert set(test.model_scope) == {'primary_frozen_models'}
    assert all('seed' not in column for column in test.columns)
    assert not any(column.startswith('Validation_') for column in test.columns)
    assert not any(column.startswith('Test_') for column in validation.columns)
    for _, row in test.iterrows():
        assert row.family == configuration(row.target)['family']
        saved = pd.read_csv(
            ROOT / f'outputs/validation/{row.target}_fixed_seed_reproduction.csv',
            float_precision='round_trip',
        )
        for name in ('R2', 'RMSE_pg_ml', 'MAE_pg_ml'):
            values = saved['Test_' + name]
            label = 'R2' if name == 'R2' else name.removesuffix('_pg_ml')
            suffix = '' if name == 'R2' else '_pg_ml'
            assert row[f'Test_{label}_mean{suffix}'] == values.mean()
            assert row[f'Test_{label}_sample_SD{suffix}'] == values.std(ddof=1)
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / 'test.csv', float_precision='round_trip'), test, check_exact=True,
    )


def test_primary_test_rejects_wrong_row_order(monkeypatch):
    original = model_comparison.split_data

    def reordered(target):
        training, test = original(target)
        return training, test.iloc[::-1]

    monkeypatch.setattr(model_comparison, 'split_data', reordered)
    with pytest.raises(AssertionError):
        primary_test_records('tnf')


def test_primary_test_requires_exact_frozen_predictions(monkeypatch):
    models = model_comparison.load_models('tnf')
    pipeline = models[0][1]
    original = pipeline.predict

    def changed(values):
        predictions = original(values)
        predictions[0] = np.nextafter(predictions[0], np.inf)
        return predictions

    monkeypatch.setattr(pipeline, 'predict', changed)
    monkeypatch.setattr(model_comparison, 'load_models', lambda target: models)
    with pytest.raises(AssertionError):
        primary_test_records('tnf')
