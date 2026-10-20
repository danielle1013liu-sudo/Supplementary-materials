import json
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from src.data import ROOT, configuration, load_data, split_data, feature_names
from src.models import make_pipeline, load_models
from src.verify import verify_validation_selection


def evidence():
    return pd.read_csv(ROOT / 'data/validation_evidence/final_seed_all20_comparisons.csv',
                       float_precision='round_trip')


def test_il10_average_selection_does_not_require_each_realisation_to_win():
    result = verify_validation_selection('il10', evidence())
    assert result['rule'] == 'three_seed_mean_validation'
    assert result['mean_Validation_winner'] == {'family': 'RandomForestRegressor', 'imputer': 'median'}
    assert [row['accepted_pipeline_wins'] for row in result['individual_Validation_winners']] == [True, False, False]


def test_tnf_still_requires_three_individual_wins():
    result = verify_validation_selection('tnf', evidence())
    assert result['rule'] == 'each_seed_validation'
    assert all(row['accepted_pipeline_wins'] for row in result['individual_Validation_winners'])
    frame = evidence()
    subset = frame.target.eq('tnf') & frame.estimator_seed.eq(configuration('tnf')['seeds'][0])
    changed = frame.index[subset & ~frame.accepted_pipeline][0]
    frame.loc[subset, 'global_same_seed_Validation_winner'] = False
    frame.loc[changed, 'global_same_seed_Validation_winner'] = True
    frame.loc[changed, 'Validation_R2'] = 1.0
    with pytest.raises(ValueError, match='every realisation'):
        verify_validation_selection('tnf', frame)


def test_il10_rejects_a_different_mean_winner():
    frame = evidence()
    subset = frame.target.eq('il10')
    changed = subset & frame.family.eq('XGBRegressor') & frame.imputer.eq('mean')
    frame.loc[subset, 'global_same_seed_Validation_winner'] = False
    frame.loc[changed, 'global_same_seed_Validation_winner'] = True
    frame.loc[changed, 'Validation_R2'] = 1.0
    with pytest.raises(ValueError, match='three-realisation mean'):
        verify_validation_selection('il10', frame)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'folds', 'parameters'])
def test_selection_rejects_incomplete_or_changed_comparisons(change):
    frame = evidence()
    index = frame.index[frame.target.eq('il10')][0]
    if change == 'missing':
        frame = frame.drop(index)
    elif change == 'duplicate':
        frame = pd.concat([frame, frame.loc[[index]]])
    elif change == 'folds':
        frame.loc[index, 'fold_hash'] = 'other folds'
    else:
        parameters = json.loads(frame.loc[index, 'parameters'])
        parameters['max_depth'] += 1
        frame.loc[index, 'parameters'] = json.dumps(parameters)
    with pytest.raises(ValueError):
        verify_validation_selection('il10', frame)


def test_approved_crosslink_values_and_fixed_split():
    frame = load_data('il10').set_index('dataset_row_id')
    expected = {**{f'IL10-{n:04d}': 'medium' for n in (3, 4, 5)},
                **{f'IL10-{n:04d}': 'high' for n in (16, 17, 18)},
                'IL10-0078': 'low'}
    assert frame.loc[list(expected), 'network_density_proxy'].to_dict() == expected
    training, test = split_data('il10')
    assert (len(training), len(test)) == (106, 27)
    assert set(expected).intersection(test.dataset_row_id) == {'IL10-0017'}


def test_saved_il10_models_use_declared_median_and_frozen_training_predictions():
    training, _ = split_data('il10')
    with np.load(ROOT / 'data/reference/accepted_predictions.npz') as reference:
        for seed, fitted in load_models('il10'):
            constructed = make_pipeline('il10', seed)
            numeric = constructed.named_steps['preprocess'].transformers[0][1].named_steps['imputer']
            assert isinstance(numeric, SimpleImputer) and numeric.strategy == 'median'
            assert constructed.named_steps['model'].max_depth == 8
            np.testing.assert_array_equal(fitted.predict(training[feature_names()]),
                                          reference[f'il10_{seed}_Training'])
