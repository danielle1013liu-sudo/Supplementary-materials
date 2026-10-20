from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID = 'dataset_row_id'
RESPONSE = 'corrected_value_pg_ml'

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def read_csv(path):
    # Keep the parser used for model fitting. Changing float_precision alters some inputs by one ULP.
    return pd.read_csv(path, keep_default_na=False, na_values=[''])

def configuration(target=None):
    values = read_json(ROOT / 'data/model_parameters.json')
    return values if target is None else values['targets'][target]

def feature_names():
    return configuration()['features']

def load_data(target):
    values = read_csv(ROOT / f'data/{target}.csv')
    assert len(values) == {'tnf': 139, 'il10': 133}[target]
    assert values[ID].is_unique
    assert values.columns[1:11].tolist() == feature_names()
    return values

def split_definition(target):
    return read_json(ROOT / f'data/splits/{target}.json')

def split_data(target):
    values = load_data(target).set_index(ID, drop=False)
    split = split_definition(target)
    train = values.loc[split['Training_IDs']].reset_index(drop=True)
    test = values.loc[split['Test_IDs']].reset_index(drop=True)
    assert set(train[ID]).isdisjoint(test[ID])
    assert set(train[ID]) | set(test[ID]) == set(values.index)
    return train, test

def cv_folds(target):
    labels = np.asarray(split_definition(target)['Validation_fold'])
    assert set(labels) == set(range(1, 11))
    assert len(labels) == len(split_data(target)[0])
    return [(np.flatnonzero(labels != fold), np.flatnonzero(labels == fold)) for fold in range(1, 11)]

def load_cases(target):
    values = read_csv(ROOT / 'data/external_cases.csv')
    return values[values.target == target].sort_values('hydrogel_group').reset_index(drop=True)

def check_inputs():
    for name, expected in read_json(ROOT / 'data/checksums.json').items():
        assert file_hash(ROOT / name) == expected, name
    for target in ('tnf', 'il10'):
        split = split_definition(target)
        for name, key in [('Training_IDs', 'Training_ID_hash'), ('Test_IDs', 'Test_ID_hash'), ('Validation_fold', 'Validation_fold_assignment_hash')]:
            assert value_hash(split[name]) == split[key], (target, key)
        split_data(target)
        cv_folds(target)
    pd.testing.assert_frame_equal(load_cases('tnf')[feature_names()], load_cases('il10')[feature_names()])

def check_environment():
    versions = read_json(ROOT / 'data/software_versions.json')
    assert platform.python_version() == versions.pop('python')
    for name, version in versions.items():
        assert importlib.metadata.version(name) == version, name

def save_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def serialise_parameters(value):
    if isinstance(value, dict):
        return {key: serialise_parameters(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [serialise_parameters(item) for item in value]
    if hasattr(value, 'get_params'):
        return {'class': type(value).__module__ + '.' + type(value).__name__, 'parameters': serialise_parameters(value.get_params(deep=False))}
    if isinstance(value, type):
        return value.__module__ + '.' + value.__name__
    if isinstance(value, np.generic):
        return serialise_parameters(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value

def record_provenance(stage, phase, scope):
    from .models import load_models
    check_inputs()
    record = {'stage': stage, 'phase': phase, 'scope': scope, 'software': read_json(ROOT / 'data/software_versions.json'),
              'inputs': [{'path': str(ROOT / name), 'sha256': file_hash(ROOT / name)} for name in read_json(ROOT / 'data/checksums.json')],
              'configuration': configuration(), 'targets': {}}
    for target in ('tnf', 'il10'):
        training, test = split_data(target)
        record['targets'][target] = {'n': len(training) + len(test), 'n_Training': len(training), 'n_Test': len(test),
                                     'split_and_folds': split_definition(target),
                                     'models': [{'seed': seed, 'parameters': serialise_parameters(model.get_params(deep=False))} for seed, model in load_models(target)]}
    save_json(record, ROOT / f'outputs/runtime_provenance/{stage}_{phase}.json')
    print(json.dumps(record, ensure_ascii=False), flush=True)
