"""Export separate archived Validation and frozen primary-model Test metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import (
    ID, RESPONSE, ROOT, check_environment, check_inputs, configuration,
    feature_names, split_data, split_definition,
)
from ..models import load_models
from ..validation import metrics


FAMILIES = ('GradientBoostingRegressor', 'XGBRegressor', 'RandomForestRegressor', 'LGBMRegressor')
REFERENCE_TOLERANCE = 1e-10
EVIDENCE = ROOT / 'data/validation_evidence'


def load_evidence():
    options = {'keep_default_na': False, 'float_precision': 'round_trip'}
    benchmarks = pd.read_csv(EVIDENCE / 'final_family_benchmarks.csv', **options)
    comparisons = pd.read_csv(EVIDENCE / 'final_seed_all20_comparisons.csv', **options)
    return benchmarks, comparisons


def matched_records(benchmark, comparisons):
    keys = ('target', 'family', 'imputer', 'config_hash')
    selected = comparisons.loc[
        comparisons[list(keys)].eq(pd.Series({key: benchmark[key] for key in keys})).all(axis=1)
    ]
    expected = configuration(benchmark['target'])['seeds']
    if (len(expected) != 3 or len(set(expected)) != 3
            or json.loads(benchmark['shared_seeds']) != expected):
        raise ValueError('Benchmark realisations differ from the released model manifest')
    if len(selected) != 3 or sorted(selected.estimator_seed.tolist()) != sorted(expected):
        raise ValueError('Expected exactly three released realisations for each benchmark configuration')
    parameters = json.loads(benchmark['parameters'])
    if any(json.loads(value) != parameters for value in selected.parameters):
        raise ValueError('Comparison parameters differ from the benchmark configuration')
    return selected


def validation_summary(benchmarks, comparisons):
    expected = {(target, family) for target in ('tnf', 'il10') for family in FAMILIES}
    actual = list(benchmarks[['target', 'family']].itertuples(index=False, name=None))
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Expected one benchmark for each of four families and both targets')
    records = []
    for _, benchmark in benchmarks.iterrows():
        selected = matched_records(benchmark, comparisons)
        metrics = selected[['Validation_R2', 'Validation_RMSE', 'Validation_MAE']]
        if not np.isfinite(metrics.to_numpy(dtype=float)).all():
            raise ValueError('Validation metrics must be finite')
        means, deviations = metrics.mean(), metrics.std(ddof=1)
        reference = benchmark[['Validation_R2_mean', 'Validation_R2_sample_SD']].to_numpy(dtype=float)
        # Use the release's CSV-reference tolerance, with no relative tolerance.
        if not np.allclose([means.Validation_R2, deviations.Validation_R2], reference,
                           atol=REFERENCE_TOLERANCE, rtol=0):
            raise ValueError('Validation R2 summary differs from the archived benchmark')
        records.append({
            'target': benchmark['target'], 'family': benchmark['family'],
            'imputer': benchmark['imputer'], 'scope': 'Validation',
            'Validation_R2_mean': means.Validation_R2,
            'Validation_R2_sample_SD': deviations.Validation_R2,
            'Validation_RMSE_mean_pg_ml': means.Validation_RMSE,
            'Validation_RMSE_sample_SD_pg_ml': deviations.Validation_RMSE,
            'Validation_MAE_mean_pg_ml': means.Validation_MAE,
            'Validation_MAE_sample_SD_pg_ml': deviations.Validation_MAE,
        })
    return pd.DataFrame(records)


def save_summary(summary, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False, lineterminator='\n')


def export_validation(output):
    summary = validation_summary(*load_evidence())
    save_summary(summary, output)
    return summary


def primary_test_records(target):
    _, test = split_data(target)
    np.testing.assert_array_equal(test[ID], split_definition(target)['Test_IDs'])
    recorded = pd.read_csv(
        ROOT / f'outputs/validation/{target}_fixed_seed_reproduction.csv',
        float_precision='round_trip',
    ).set_index('seed')
    records = []
    with np.load(ROOT / 'data/reference/accepted_predictions.npz') as reference:
        for seed, pipeline in load_models(target):
            predicted = pipeline.predict(test[feature_names()])
            np.testing.assert_array_equal(predicted, reference[f'{target}_{seed}_Test'])
            measured = metrics(test[RESPONSE], predicted)
            for name, value in measured.items():
                if value != recorded.loc[seed, 'Test_' + name]:
                    raise ValueError(f'Primary Test {name} differs from the frozen reference')
            records.append(measured)
    return pd.DataFrame(records)


def primary_test_summary():
    benchmarks, _ = load_evidence()
    records = []
    for target in ('tnf', 'il10'):
        family = configuration(target)['family']
        benchmark = benchmarks.loc[benchmarks.target.eq(target) & benchmarks.family.eq(family)]
        if len(benchmark) != 1:
            raise ValueError('Expected one archived benchmark for the primary model')
        measured = primary_test_records(target)
        means, deviations = measured.mean(), measured.std(ddof=1)
        records.append({
            'target': target, 'family': family, 'imputer': benchmark.imputer.item(),
            'scope': 'Test', 'model_scope': 'primary_frozen_models',
            'Test_R2_mean': means.R2, 'Test_R2_sample_SD': deviations.R2,
            'Test_RMSE_mean_pg_ml': means.RMSE_pg_ml,
            'Test_RMSE_sample_SD_pg_ml': deviations.RMSE_pg_ml,
            'Test_MAE_mean_pg_ml': means.MAE_pg_ml,
            'Test_MAE_sample_SD_pg_ml': deviations.MAE_pg_ml,
        })
    return pd.DataFrame(records)


def export_primary_test(output):
    summary = primary_test_summary()
    save_summary(summary, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/model_comparison')
    args = parser.parse_args()
    check_environment()
    check_inputs()
    validation = export_validation(args.output_dir / 'validation.csv')
    test = export_primary_test(args.output_dir / 'test.csv')
    print(f'Exported {len(validation)} archived Validation and {len(test)} primary Test summaries to {args.output_dir}')


if __name__ == '__main__':
    main()
