from .data import record_provenance
import argparse
import importlib.metadata
import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .data import ROOT, RESPONSE, check_environment, check_inputs, configuration, read_json, split_data, save_csv
from .models import make_preprocessor
from .validation import metrics, validation_predictions

IMPUTERS = ['median', 'mean', 'iterative_bayesian_ridge', 'scaled_knn_3', 'scaled_knn_5']

class ScaledKNN(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        values = np.asarray(X, float)
        assert not np.isnan(values).all(axis=0).any()
        self.scaler_ = StandardScaler().fit(values)
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors, weights='distance').fit(self.scaler_.transform(values))
        return self

    def transform(self, X):
        values = np.asarray(X, float)
        result = self.scaler_.inverse_transform(self.imputer_.transform(self.scaler_.transform(values)))
        result[~np.isnan(values)] = values[~np.isnan(values)]
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)

def grid_pipeline(target, family, imputer, parameters, seed):
    assert seed in configuration(target)['seeds']
    preprocess = make_preprocessor(target)
    if imputer in ('mean', 'median'):
        numeric = SimpleImputer(strategy=imputer)
    elif imputer == 'iterative_bayesian_ridge':
        numeric = IterativeImputer(estimator=BayesianRidge(), max_iter=20, sample_posterior=False, random_state=42)
    else:
        numeric = ScaledKNN(3 if imputer == 'scaled_knn_3' else 5)
    preprocess.set_params(numerical__imputer=numeric)
    parameters = {**parameters, 'random_state': seed}
    if family == 'XGBRegressor':
        from xgboost import XGBRegressor
        parameters.update(n_jobs=1, verbosity=0)
        if parameters.get('missing') is None:
            parameters['missing'] = np.nan
        estimator = XGBRegressor(**parameters)
    elif family == 'LGBMRegressor':
        from lightgbm import LGBMRegressor
        parameters.update(bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed, drop_seed=seed,
                          deterministic=True, force_col_wise=True, device_type='cpu', n_jobs=1, verbosity=-1)
        estimator = LGBMRegressor(**parameters)
    elif family == 'RandomForestRegressor':
        parameters['n_jobs'] = 1
        estimator = RandomForestRegressor(**parameters)
    else:
        assert family == 'GradientBoostingRegressor'
        estimator = GradientBoostingRegressor(**parameters)
    return Pipeline([('preprocess', preprocess), ('model', estimator)])

def compare_grid(target):
    training, _ = split_data(target)
    grid = read_json(ROOT / 'data/local_grids.json')[target]
    records = []
    for seed in configuration(target)['seeds']:
        for family, candidates in grid.items():
            for imputer in IMPUTERS:
                for candidate in candidates:
                    pipeline = grid_pipeline(target, family, imputer, candidate['parameters'], seed)
                    predictions = validation_predictions(target, pipeline, training)
                    records.append({'target': target, 'fixed_seed': seed, 'family': family, 'imputer': imputer,
                                    'candidate': candidate['label'], 'parameters': json.dumps(candidate['parameters'], sort_keys=True),
                                    **metrics(training[RESPONSE], predictions)})
    result = pd.DataFrame(records).sort_values(['fixed_seed', 'R2', 'RMSE_pg_ml', 'MAE_pg_ml'], ascending=[True, False, True, True], kind='stable')
    result['same_seed_validation_rank'] = result.groupby('fixed_seed', sort=False).cumcount() + 1
    save_csv(result, ROOT / f'outputs/optional_grid_search/{target}_all_candidates.csv')
    best = result.drop_duplicates(['fixed_seed', 'family', 'imputer'])
    save_csv(best, ROOT / f'outputs/optional_grid_search/{target}_best_per_family_imputer.csv')
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['tnf', 'il10'], required=True)
    arguments = parser.parse_args()
    check_environment()
    record_provenance('grid_search', 'before', 'Optional Training-only local-grid comparison; shared saved folds; accepted fixed seeds only.')
    for package, version in read_json(ROOT / 'data/grid_software_versions.json').items():
        assert importlib.metadata.version(package) == version
    compare_grid(arguments.target)
    record_provenance('grid_search', 'after', 'Optional Training-only local-grid comparison; shared saved folds; accepted fixed seeds only.')
    print('Training-only local-grid comparison exported. Accepted model parameters and fixed seeds are unchanged.')
