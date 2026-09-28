import pickle
import zlib
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from .data import ROOT, configuration, read_json, file_hash

def numeric_values(values):
    if isinstance(values, dict):
        return {key: numeric_values(value) for key, value in values.items()}
    if isinstance(values, list):
        return [numeric_values(value) for value in values]
    if isinstance(values, str) and values in ('NaN', 'Infinity', '-Infinity'):
        return float(values)
    return values

def make_preprocessor(target):
    settings = configuration(target)
    numeric = numeric_values(settings['numeric_imputer'])
    if target == 'tnf':
        imputer = SimpleImputer(**numeric)
    else:
        numeric['estimator'] = BayesianRidge(**numeric['estimator'])
        imputer = IterativeImputer(**numeric)
    encoder = settings['one_hot_encoder'].copy()
    encoder['dtype'] = np.float64
    categories = Pipeline([
        ('imputer', SimpleImputer(**numeric_values(settings['categorical_imputer']))),
        ('onehot', OneHotEncoder(**encoder))
    ])
    return ColumnTransformer([
        ('numerical', Pipeline([('imputer', imputer)]), settings['numeric_columns']),
        ('categorical', categories, settings['categorical_columns'])
    ], **settings['column_transformer'])

def make_pipeline(target, seed, parameters=None):
    settings = configuration(target)
    assert seed in settings['seeds']
    arguments = settings['parameters'].copy() if parameters is None else parameters.copy()
    arguments['random_state'] = seed
    estimator = GradientBoostingRegressor(**arguments) if target == 'tnf' else RandomForestRegressor(**arguments)
    return Pipeline([('preprocess', make_preprocessor(target)), ('model', estimator)])

def load_models(target):
    models = []
    for item in read_json(ROOT / 'models/manifest.json'):
        if item['target'] != target:
            continue
        path = ROOT / item['path']
        assert file_hash(path) == item['sha256']
        pipeline = pickle.loads(zlib.decompress(path.read_bytes()))
        assert pipeline.named_steps['model'].get_params() == make_pipeline(target, item['seed']).named_steps['model'].get_params()
        models.append((item['seed'], pipeline))
    assert [seed for seed, _ in models] == configuration(target)['seeds']
    return models

def save_model(pipeline, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zlib.compress(pickle.dumps(pipeline, protocol=5), 3))
