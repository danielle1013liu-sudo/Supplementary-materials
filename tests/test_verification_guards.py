import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src import public_predict
from src.pareto.common import FEATURES
from src.pareto.verification import SHAP_TOLERANCE, verify_profile_counts, verify_shap_additivity


@pytest.fixture
def prediction_paths(tmp_path, monkeypatch):
    repository = tmp_path / 'repository'
    repository.mkdir()
    source = tmp_path / 'input.csv'
    source.write_text('dataset_row_id,feature\nexample,2\n')
    monkeypatch.setattr(public_predict, 'ROOT', repository)
    return source, repository


@pytest.mark.parametrize('directory', ['data', 'models', 'src', 'docs', 'tests', '.git', 'configs'])
def test_prediction_rejects_protected_directories(prediction_paths, directory):
    source, repository = prediction_paths
    with pytest.raises(ValueError, match='protected repository path'):
        public_predict.validate_output_path(source, repository / directory / 'predictions.csv')


@pytest.mark.parametrize('filename', ['results/frozen.csv', 'outputs/frozen.csv', 'requirements.txt'])
def test_prediction_preserves_existing_repository_files(prediction_paths, filename):
    source, repository = prediction_paths
    destination = repository / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('frozen')
    with pytest.raises(ValueError, match='existing repository file'):
        public_predict.validate_output_path(source, destination)
    assert destination.read_text() == 'frozen'


def test_prediction_rejects_input_alias_before_environment_or_models(prediction_paths, monkeypatch):
    source, repository = prediction_paths
    destination = repository / 'alias.csv'
    destination.write_text('alias')
    monkeypatch.setattr(Path, 'samefile', lambda self, other: self == destination and other == source)
    original = source.read_bytes()
    monkeypatch.setattr(sys, 'argv', ['public_predict', str(source), '--target', 'tnf', '--output', str(destination)])
    environment = Mock(side_effect=AssertionError('Output validation must run first'))
    models = Mock(side_effect=AssertionError('Models must not load'))
    monkeypatch.setattr(public_predict, 'check_environment', environment)
    monkeypatch.setattr(public_predict, 'load_models', models)
    with pytest.raises(ValueError, match='alias the input'):
        public_predict.main()
    environment.assert_not_called()
    models.assert_not_called()
    assert source.read_bytes() == original


def test_prediction_rejects_external_hardlink_to_a_repository_file(prediction_paths, monkeypatch):
    source, repository = prediction_paths
    protected = repository / 'results' / 'frozen.csv'
    protected.parent.mkdir()
    protected.write_text('frozen')
    destination = source.parent / 'alias.csv'
    destination.write_text('alias')
    original_stat = Path.stat

    def alias_stat(path, **kwargs):
        result = original_stat(path, **kwargs)
        return SimpleNamespace(st_nlink=2, st_mode=result.st_mode) if path == destination else result

    monkeypatch.setattr(Path, 'stat', alias_stat)
    monkeypatch.setattr(Path, 'samefile', lambda self, other: self == destination and other == protected)
    with pytest.raises(ValueError, match='alias a protected repository file'):
        public_predict.validate_output_path(source, destination)
    assert protected.read_text() == 'frozen'


def test_prediction_rejects_resolved_input_path(prediction_paths):
    source, repository = prediction_paths
    with pytest.raises(ValueError, match='alias the input'):
        public_predict.validate_output_path(source, repository / '..' / source.name)


@pytest.mark.parametrize('directory', ['results', 'outputs', 'tmp'])
def test_prediction_allows_new_output_files(prediction_paths, directory):
    source, repository = prediction_paths
    destination = repository / directory / 'predictions.csv'
    assert public_predict.validate_output_path(source, destination) == destination
    assert not destination.exists()


def test_prediction_allows_reusing_temporary_and_external_outputs(prediction_paths):
    source, repository = prediction_paths
    for destination in [repository / 'tmp' / 'predictions.csv', source.parent / 'predictions.csv']:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text('previous output')
        assert public_predict.validate_output_path(source, destination) == destination


def test_prediction_runs_for_a_new_result_path(prediction_paths, monkeypatch):
    source, repository = prediction_paths
    destination = repository / 'results' / 'predictions.csv'
    monkeypatch.setattr(sys, 'argv', ['public_predict', str(source), '--target', 'tnf', '--output', str(destination)])
    monkeypatch.setattr(public_predict, 'check_environment', lambda: None)
    monkeypatch.setattr(public_predict, 'check_inputs', lambda: None)
    monkeypatch.setattr(public_predict, 'decode_public_input', lambda frame, version: frame)
    monkeypatch.setattr(public_predict, 'feature_names', lambda: ['feature'])
    pipeline = Mock()
    pipeline.predict.return_value = np.array([3.0])
    monkeypatch.setattr(public_predict, 'load_models', lambda target: [(3089, pipeline)])
    public_predict.main()
    pipeline.predict.assert_called_once()
    result = pd.read_csv(destination)
    assert result.dataset_row_id.tolist() == ['example']
    assert result.prediction_pg_ml.tolist() == [3.0]


@pytest.fixture
def shap_rows():
    values = pd.DataFrame({
        'target': ['tnf'] * len(FEATURES), 'candidate_id': ['profile'] * len(FEATURES),
        'model_realisation': ['3089'] * len(FEATURES), 'descriptor_name': FEATURES,
        'expected_value_pg_ml': [5.0] * len(FEATURES),
        'prediction_pg_ml': [15.0] * len(FEATURES),
        'grouped_SHAP_pg_ml': [1.0] * len(FEATURES),
    })
    recorded = pd.DataFrame([{'target': 'tnf', 'candidate_id': 'profile',
                             'model_realisation': '3089', 'additivity_residual_pg_ml': 0.0}])
    return values, recorded


def test_shap_residual_is_recomputed_and_bounded(shap_rows):
    values, recorded = shap_rows
    assert verify_shap_additivity(values, recorded) == 0
    values['prediction_pg_ml'] += 2 * SHAP_TOLERANCE
    with pytest.raises(AssertionError, match='SHAP additivity failed'):
        verify_shap_additivity(values, recorded)


def test_shap_rejects_a_forged_residual_summary(shap_rows):
    values, recorded = shap_rows
    recorded['additivity_residual_pg_ml'] = 1.0
    with pytest.raises(AssertionError):
        verify_shap_additivity(values, recorded)


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -np.inf])
def test_shap_rejects_nonfinite_contributions(shap_rows, invalid):
    values, recorded = shap_rows
    values.loc[0, 'grouped_SHAP_pg_ml'] = invalid
    with pytest.raises(ValueError, match='finite'):
        verify_shap_additivity(values, recorded)


def test_shap_rejects_incomplete_descriptors(shap_rows):
    values, recorded = shap_rows
    with pytest.raises(ValueError, match='Incomplete SHAP descriptors'):
        verify_shap_additivity(values.iloc[:-1], recorded)


def test_shap_rejects_inconsistent_repeated_predictions(shap_rows):
    values, recorded = shap_rows
    values.loc[0, 'prediction_pg_ml'] += 1
    with pytest.raises(AssertionError):
        verify_shap_additivity(values, recorded)


def test_pareto_counts_distinguish_profiles_from_objective_locations():
    profiles = pd.DataFrame({'tnf_mean_pg_ml': [1., 1., 2.], 'il10_mean_pg_ml': [3., 3., 4.]})
    expected = {'final_profiles': 3, 'final_locations': 2}
    assert verify_profile_counts(expected, profiles) == expected
    for key in expected:
        altered = {**expected, key: 99}
        with pytest.raises(ValueError, match=key):
            verify_profile_counts(altered, profiles)
