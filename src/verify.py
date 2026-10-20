"""Check frozen predictions, split membership and target-specific selection rules."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from .data import (
    ROOT, ID, RESPONSE, check_environment, configuration, feature_names, load_data,
    read_json, record_provenance, save_csv, save_json, serialise_parameters,
    split_data, split_definition,
)
from .models import load_models, make_pipeline
from .predict import export_predictions
from .validation import metrics


def verify_validation_selection(target, comparisons):
    """TNF retains individual wins; revised IL-10 uses the three-run mean."""
    settings = configuration(target)
    expected_rule = {'tnf': 'each_seed_validation', 'il10': 'three_seed_mean_validation'}[target]
    if settings['selection_rule'] != expected_rule:
        raise ValueError(f'{target}: the released selection rule changed')
    selected = comparisons.loc[comparisons.target.eq(target)].copy()
    locked = read_json(ROOT / 'data/locked_comparators.json')[target]
    keys = ['family', 'imputer']
    expected = {(row['family'], row['imputer']): row for row in locked}
    if len(expected) != 20 or len(locked) != 20 or len(selected) != 60:
        raise ValueError('Expected 20 fixed configurations at each of three realisations')
    if set(selected.estimator_seed) != set(settings['seeds']):
        raise ValueError('Comparator realisations differ from the released model')
    score_columns = ['Validation_R2', 'Validation_RMSE', 'Validation_MAE']
    if not np.isfinite(selected[score_columns].to_numpy(dtype=float)).all():
        raise ValueError('Comparator metrics must be finite')
    primary = (settings['family'], settings['imputer'])
    winners = []
    for seed in settings['seeds']:
        table = selected.loc[selected.estimator_seed.eq(seed)]
        identities = list(table[keys].itertuples(index=False, name=None))
        if len(identities) != 20 or set(identities) != set(expected):
            raise ValueError('Missing or duplicated same-realisation comparator')
        for row in table.itertuples(index=False):
            identity = (row.family, row.imputer)
            fixed = expected[identity]
            if row.config_hash != fixed['config_hash'] or json.loads(row.parameters) != fixed['parameters']:
                raise ValueError('Comparator parameters differ from the locked configuration')
            # The TNF archive uses its original fold identifier; verify_results
            # independently reconstructs the actual saved split and fold labels.
            if row.fold_hash != settings['validation_evidence_fold_hash']:
                raise ValueError('Comparator folds differ from the released folds')
            if bool(row.accepted_pipeline) != (identity == primary):
                raise ValueError('Accepted-pipeline marker differs from the released model')
        ordered = table.sort_values(score_columns + keys, ascending=[False, True, True, True, True], kind='stable')
        winner = ordered.iloc[0]
        actual = (winner.family, winner.imputer)
        flags = table.loc[table.global_same_seed_Validation_winner, keys]
        if list(flags.itertuples(index=False, name=None)) != [actual]:
            raise ValueError('Same-realisation winner flag does not match Validation scores')
        winners.append({'seed': seed, 'family': actual[0], 'imputer': actual[1],
                        'accepted_pipeline_wins': actual == primary})
        if expected_rule == 'each_seed_validation' and actual != primary:
            raise ValueError('TNF requires the accepted pipeline to win at every realisation')
    means = selected.groupby(keys, sort=False)[score_columns].mean().reset_index()
    means = means.sort_values(score_columns + keys, ascending=[False, True, True, True, True], kind='stable')
    winner = means.iloc[0]
    if expected_rule == 'three_seed_mean_validation' and (winner.family, winner.imputer) != primary:
        raise ValueError('IL10 requires the accepted pipeline to win by three-realisation mean Validation')
    return {'rule': expected_rule, 'configuration_count': 20, 'realisations': settings['seeds'],
            'mean_Validation_winner': {'family': winner.family, 'imputer': winner.imputer},
            'individual_Validation_winners': winners}


def verify_results():
    export_predictions()
    reference = np.load(ROOT / 'data/reference/accepted_predictions.npz')
    comparisons = pd.read_csv(ROOT / 'data/validation_evidence/final_seed_all20_comparisons.csv', float_precision='round_trip')
    scores, checks, selection = [], [], {}
    for target in ('tnf', 'il10'):
        selection[target] = verify_validation_selection(target, comparisons)
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
            np.testing.assert_array_equal(selected.observed_pg_ml, training[RESPONSE])
            np.testing.assert_array_equal(selected.prediction_pg_ml, reference[f'{target}_{seed}_OOF'])
            np.testing.assert_array_equal(pipeline.predict(training[feature_names()]), reference[f'{target}_{seed}_Training'])
            validation_score = metrics(training[RESPONSE], reference[f'{target}_{seed}_OOF'])
            test_score = metrics(test[RESPONSE], reference[f'{target}_{seed}_Test'])
            training_score = metrics(training[RESPONSE], reference[f'{target}_{seed}_Training'])
            for scope, measured in [('Training', training_score), ('Validation', validation_score), ('Test', test_score)]:
                for name, value in measured.items():
                    assert value == recorded.loc[seed, scope + '_' + name], (target, seed, scope, name)
            primary_row = comparisons.loc[
                comparisons.target.eq(target) & comparisons.estimator_seed.eq(seed) & comparisons.accepted_pipeline
            ].iloc[0]
            np.testing.assert_allclose(
                [primary_row.Validation_R2, primary_row.Validation_RMSE, primary_row.Validation_MAE],
                [validation_score['R2'], validation_score['RMSE_pg_ml'], validation_score['MAE_pg_ml']],
                rtol=0, atol=1e-10,
            )
            current.append({'seed': seed, 'Training_R2': training_score['R2'],
                            'Validation_R2': validation_score['R2'], 'Test_R2': test_score['R2']})
            checks.append({'target': target, 'seed': seed, 'Training_n': len(training), 'Test_n': len(test),
                           'parameters_equal_saved_pipeline': True, 'stored_OOF_exact': True,
                           'fresh_Training_Test_case_predictions_exact': True})
        current = pd.DataFrame(current)
        for scope in ['Training', 'Validation', 'Test']:
            scores.append({'target': target, 'scope': scope, 'R2_mean': current[scope + '_R2'].mean(),
                           'R2_sample_SD': current[scope + '_R2'].std(ddof=1),
                           'same_three_seeds': json.dumps(configuration(target)['seeds'])})
    save_csv(pd.DataFrame(scores), ROOT / 'outputs/validation/panel_summary.csv')
    save_json({
        'input_and_model_hashes': 'PASS', 'split_and_CV_seed_reconstruction': 'PASS',
        'saved_and_constructed_pipeline_parameters': 'exact',
        'six_saved_Training_Test_case_predictions': 'exact',
        'stored_pooled_OOF_predictions_and_metrics': 'exact',
        'Validation_selection': selection,
        'predictive_model_fitting_in_this_verification': False, 'checks': checks,
    }, ROOT / 'outputs/verification.json')


if __name__ == '__main__':
    check_environment()
    record_provenance('verify', 'before', 'Accepted fixed models, exact Training/Test/CV definitions and stored Validation evidence; no model fitting.')
    verify_results()
    record_provenance('verify', 'after', 'Accepted fixed models, exact Training/Test/CV definitions and stored Validation evidence; no model fitting.')
    print('Accepted splits, folds, pipeline parameters, Validation rules and predictions verified.')
