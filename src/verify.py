import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from .data import ROOT, ID, RESPONSE, check_environment, configuration, load_data, record_provenance, save_csv, save_json, serialise_parameters, split_data, split_definition
from .models import load_models, make_pipeline
from .predict import export_predictions
from .validation import metrics

def verify_results():
    export_predictions()
    reference = np.load(ROOT / 'data/reference/accepted_predictions.npz')
    comparisons = pd.read_csv(ROOT / 'data/validation_evidence/final_seed_all20_comparisons.csv', float_precision='round_trip')
    scores = []
    checks = []
    for target in ('tnf', 'il10'):
        definition = split_definition(target)
        values = load_data(target)
        training_ids, test_ids = train_test_split(values[ID], test_size=.2, random_state=definition['split_seed'], shuffle=True)
        assert training_ids.tolist() == definition['Training_IDs']
        assert test_ids.tolist() == definition['Test_IDs']
        labels = np.zeros(len(training_ids), int)
        for fold, (_, validation) in enumerate(KFold(10, shuffle=True, random_state=definition['cv_seed']).split(training_ids), start=1):
            labels[validation] = fold
        np.testing.assert_array_equal(labels, definition['Validation_fold'])
        training, test = split_data(target)
        oof = pd.read_csv(ROOT / f'outputs/validation/{target}_OOF_predictions.csv', float_precision='round_trip')
        recorded = pd.read_csv(ROOT / f'outputs/validation/{target}_fixed_seed_reproduction.csv', float_precision='round_trip').set_index('seed')
        current = []
        for seed, pipeline in load_models(target):
            constructed = make_pipeline(target, seed)
            assert serialise_parameters(pipeline.get_params(deep=False)) == serialise_parameters(constructed.get_params(deep=False))
            selected = oof[oof.seed == seed]
            np.testing.assert_array_equal(selected[ID], definition['Training_IDs'])
            np.testing.assert_array_equal(selected.Validation_fold, definition['Validation_fold'])
            np.testing.assert_array_equal(selected.prediction_pg_ml, reference[f'{target}_{seed}_OOF'])
            validation_score = metrics(training[RESPONSE], reference[f'{target}_{seed}_OOF'])
            test_score = metrics(test[RESPONSE], reference[f'{target}_{seed}_Test'])
            for scope, measured in [('Validation', validation_score), ('Test', test_score)]:
                for name, value in measured.items():
                    assert value == recorded.loc[seed, scope + '_' + name], (target, seed, scope, name)
            table = comparisons[(comparisons.target == target) & (comparisons.estimator_seed == seed)]
            assert len(table) == 20 and len(table[['family', 'imputer']].drop_duplicates()) == 20
            winner = table.sort_values(['Validation_R2', 'Validation_RMSE', 'Validation_MAE'], ascending=[False, True, True], kind='stable').iloc[0]
            assert bool(winner.accepted_pipeline)
            assert winner.family == configuration(target)['family']
            current.append({'seed': seed, 'Validation_R2': validation_score['R2'], 'Test_R2': test_score['R2']})
            checks.append({'target': target, 'seed': seed, 'Training_n': len(training), 'Test_n': len(test), 'parameters_equal_saved_pipeline': True, 'stored_OOF_exact': True, 'fresh_Test_case_predictions_exact': True, 'all20_Validation_winner': True})
        current = pd.DataFrame(current)
        for scope in ['Validation', 'Test']:
            scores.append({'target': target, 'scope': scope, 'R2_mean': current[scope + '_R2'].mean(), 'R2_sample_SD': current[scope + '_R2'].std(ddof=1), 'same_three_seeds': json.dumps(configuration(target)['seeds'])})
    save_csv(pd.DataFrame(scores), ROOT / 'outputs/validation/panel_summary.csv')
    save_json({
        'input_and_model_hashes': 'PASS',
        'split_and_CV_seed_reconstruction': 'PASS',
        'saved_and_constructed_pipeline_parameters': 'exact',
        'six_saved_Test_case_predictions': 'exact',
        'stored_pooled_OOF_predictions_and_metrics': 'exact',
        'same_seed_all20_Validation_winners': 'PASS',
        'predictive_model_fitting_in_this_verification': False,
        'checks': checks,
    }, ROOT / 'outputs/verification.json')

if __name__ == '__main__':
    check_environment()
    record_provenance('verify', 'before', 'Accepted fixed models, exact Training/Test/CV definitions and stored Validation evidence; no model fitting.')
    verify_results()
    record_provenance('verify', 'after', 'Accepted fixed models, exact Training/Test/CV definitions and stored Validation evidence; no model fitting.')
    print('Accepted splits, folds, pipeline parameters, Validation evidence, Test and case predictions verified.')
