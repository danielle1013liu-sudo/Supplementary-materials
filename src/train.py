from .data import record_provenance
import argparse
import numpy as np
import pandas as pd
from .data import ROOT, ID, RESPONSE, check_environment, check_inputs, configuration, feature_names, load_cases, split_data, split_definition, save_csv, save_json
from .models import make_pipeline, save_model, load_models
from .validation import metrics, validation_predictions

def train_target(target):
    training, test = split_data(target)
    cases = load_cases(target)
    reference = np.load(ROOT / 'data/reference/accepted_predictions.npz')
    fitted_reference = dict(load_models(target))
    records = []
    validation_rows = []
    for seed in configuration(target)['seeds']:
        pipeline = make_pipeline(target, seed)
        validation = validation_predictions(target, pipeline, training)
        np.testing.assert_array_equal(validation, reference[f'{target}_{seed}_OOF'])
        pipeline.fit(training[feature_names()], training[RESPONSE])
        combined = pd.concat([test[feature_names()], cases[feature_names()]], ignore_index=True)
        predicted = pipeline.predict(combined)
        np.testing.assert_array_equal(predicted[:-2], reference[f'{target}_{seed}_Test'])
        np.testing.assert_array_equal(predicted[-2:], reference[f'{target}_{seed}_C4'])
        np.testing.assert_array_equal(pipeline.predict(training[feature_names()]), fitted_reference[seed].predict(training[feature_names()]))
        record = {'target': target, 'seed': seed, 'fit_n': len(training), 'fit_ID_hash': split_definition(target)['Training_ID_hash'],
                  'OOF_exact': True, 'Test_exact': True, 'case_exact': True, 'Training_predictions_exact': True}
        record.update({f'Validation_{key}': value for key, value in metrics(training[RESPONSE], validation).items()})
        record.update({f'Test_{key}': value for key, value in metrics(test[RESPONSE], predicted[:-2]).items()})
        records.append(record)
        for row_id, fold, observed, value in zip(training[ID], split_definition(target)['Validation_fold'], training[RESPONSE], validation):
            validation_rows.append({'target': target, 'seed': seed, ID: row_id, 'Validation_fold': fold, 'observed_pg_ml': observed, 'prediction_pg_ml': value})
        save_model(pipeline, ROOT / f'outputs/retrained_models/{target}_{seed}.pklz')
    save_csv(pd.DataFrame(records), ROOT / f'outputs/validation/{target}_fixed_seed_reproduction.csv')
    save_csv(pd.DataFrame(validation_rows), ROOT / f'outputs/validation/{target}_OOF_predictions.csv')
    return records

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['tnf', 'il10', 'both'], default='both')
    arguments = parser.parse_args()
    check_environment()
    record_provenance('train', 'before', 'Only saved Training IDs; saved ten-fold OOF fits use fold Training rows only; six accepted seeds.')
    targets = ['tnf', 'il10'] if arguments.target == 'both' else [arguments.target]
    results = [row for target in targets for row in train_target(target)]
    save_json({'fixed_seed_reproduction': results, 'fit_scope': 'Exact saved Training IDs only', 'prediction_equality': 'NumPy array_equal, no tolerance'}, ROOT / 'outputs/validation/reproduction.json')
    record_provenance('train', 'after', 'Only saved Training IDs; saved ten-fold OOF fits use fold Training rows only; six accepted seeds.')
    print('Fixed-seed Training, ten-fold OOF, Test and case predictions reproduced exactly.')
