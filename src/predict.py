from .data import record_provenance
import numpy as np
import pandas as pd
from .data import ROOT, ID, RESPONSE, check_environment, check_inputs, feature_names, load_cases, load_data, split_data, save_csv
from .models import load_models
from .validation import metrics

def export_predictions():
    reference = np.load(ROOT / 'data/reference/accepted_predictions.npz')
    records = []
    scores = []
    for target in ('tnf', 'il10'):
        training, test = split_data(target)
        cases = load_cases(target)
        all_rows = load_data(target)
        for seed, pipeline in load_models(target):
            combined = pd.concat([test[feature_names()], cases[feature_names()]], ignore_index=True)
            predictions = pipeline.predict(combined)
            np.testing.assert_array_equal(predictions[:-2], reference[f'{target}_{seed}_Test'])
            np.testing.assert_array_equal(predictions[-2:], reference[f'{target}_{seed}_C4'])
            score = metrics(test[RESPONSE], predictions[:-2])
            scores.append({'target': target, 'seed': seed, **score})
            values = pipeline.predict(all_rows[feature_names()])
            for row, value in zip(all_rows.itertuples(index=False), values):
                row_id = getattr(row, ID)
                records.append({'target': target, 'seed': seed, ID: row_id,
                                'scope': 'Training-in-sample' if row_id in set(training[ID]) else 'Test-out-of-fit',
                                'observed_pg_ml': getattr(row, RESPONSE), 'prediction_pg_ml': value})
            for group, value, observed in zip((2, 4), predictions[-2:], (612, 488.963) if target == 'tnf' else (155, 188.06)):
                records.append({'target': target, 'seed': seed, ID: f'C4_Group{group}', 'scope': 'external',
                                'observed_pg_ml': observed, 'prediction_pg_ml': value})
    predictions = pd.DataFrame(records)
    save_csv(predictions, ROOT / 'outputs/predictions/by_seed.csv')
    summary = predictions.groupby(['target', ID, 'scope', 'observed_pg_ml'], sort=False).prediction_pg_ml.agg(['mean', 'std']).reset_index()
    summary = summary.rename(columns={'mean': 'prediction_mean_pg_ml', 'std': 'prediction_sample_SD_pg_ml'})
    save_csv(summary, ROOT / 'outputs/predictions/summary.csv')
    save_csv(pd.DataFrame(scores), ROOT / 'outputs/predictions/test_metrics.csv')
    return predictions

if __name__ == '__main__':
    check_environment()
    record_provenance('predict', 'before', 'Frozen Training-only models; Training-in-sample, Test-out-of-fit and external cases.')
    export_predictions()
    record_provenance('predict', 'after', 'Frozen Training-only models; Training-in-sample, Test-out-of-fit and external cases.')
    print('Accepted Test and external-case predictions reproduced exactly for all six saved models.')
