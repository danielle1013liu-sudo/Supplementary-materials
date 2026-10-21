import numpy as np
import pandas as pd
import pytest

from src.data import read_json, ROOT
from src.hydrogel_immune_ml import application_runtime as runtime
from src.hydrogel_immune_ml.inference import load_resources


def test_cloud_accepts_python_patch_with_exact_packages(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.platform, 'python_version', lambda: '3.13.99')
    runtime.check_cloud_environment(tmp_path)
    table = pd.read_csv(tmp_path / 'environment.csv')
    assert table.passed.all()
    assert table.loc[table.package.eq('python'), 'actual_version'].iloc[0] == '3.13.99'


@pytest.mark.parametrize('version', ['3.12.9', '3.14.0'])
def test_cloud_rejects_other_python_minor(monkeypatch, tmp_path, version):
    monkeypatch.setattr(runtime.platform, 'python_version', lambda: version)
    with pytest.raises(ValueError, match='python'):
        runtime.check_cloud_environment(tmp_path)
    assert not pd.read_csv(tmp_path / 'environment.csv').passed.all()


def test_cloud_rejects_changed_scientific_package(monkeypatch, tmp_path):
    original = runtime.importlib.metadata.version
    monkeypatch.setattr(runtime.importlib.metadata, 'version',
                        lambda name: '0.0.0' if name == 'scikit-learn' else original(name))
    with pytest.raises(ValueError, match='scikit-learn'):
        runtime.check_cloud_environment(tmp_path)


def test_cloud_rejects_missing_scientific_package(monkeypatch, tmp_path):
    original = runtime.importlib.metadata.version

    def version(name):
        if name == 'shap':
            raise runtime.importlib.metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(runtime.importlib.metadata, 'version', version)
    with pytest.raises(ValueError, match='shap'):
        runtime.check_cloud_environment(tmp_path)


def test_unknown_runtime_is_rejected():
    with pytest.raises(ValueError, match='Unknown application runtime'):
        load_resources(runtime='unchecked')


def test_reproduction_still_rejects_other_python_patch(monkeypatch):
    monkeypatch.setattr(runtime.platform, 'python_version', lambda: '3.13.99')
    with pytest.raises(AssertionError):
        load_resources()


def test_cloud_loads_saved_models_and_replays_without_fitting(monkeypatch):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    def forbid(*args, **kwargs):
        raise AssertionError('The cloud application must not fit models')

    for estimator in (Pipeline, ColumnTransformer):
        monkeypatch.setattr(estimator, 'fit', forbid)
        monkeypatch.setattr(estimator, 'fit_transform', forbid)
    resources = load_resources(runtime='community-cloud')
    checks = pd.read_csv(ROOT / 'results/runtime/prediction_replay_checks.csv')
    assert len(checks) == len(read_json(ROOT / 'models/manifest.json')) * 3
    assert checks.passed.all()
    assert set(resources) == {'tnf', 'il10'}


def test_cloud_rejects_prediction_mismatch(monkeypatch, tmp_path):
    resources = load_resources()
    pipeline = resources['tnf']['models'][0][1]
    original = pipeline.predict
    monkeypatch.setattr(pipeline, 'predict', lambda frame: np.nextafter(original(frame), np.inf))
    with pytest.raises(ValueError, match='predictions differ'):
        runtime.verify_cloud_predictions(resources, tmp_path)
    checks = pd.read_csv(tmp_path / 'prediction_replay_checks.csv')
    assert not checks.passed.all()
    assert (tmp_path / 'prediction_replay.csv').is_file()
