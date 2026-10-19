"""Verify publication-code changes against frozen models and numerical evidence."""

import argparse
import gzip
import shutil
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits

from .. import predict, verify
from ..case_replay import run as replay_cases
from ..data import (
    ROOT, check_environment, check_inputs, feature_names, file_hash,
    load_cases, load_data, read_csv, read_json, save_csv, split_definition,
)
from ..models import load_models
from ..pareto import verification
from ..pareto.common import CAT, DATA, FEATURES, SCHEMA, SEEDS, csv, table, training
from ..pareto.generation import verify_models
from .public_encoding import PUBLIC_VERSION, decode_public_input, encode_public_input


@contextmanager
def forbid_fitting():
    """Fail on any Python fit call during frozen-object verification."""
    previous, thread_previous = sys.getprofile(), threading.getprofile()

    def guard(frame, event, argument):
        if event == 'call' and frame.f_code.co_name in ('fit', 'fit_transform'):
            raise RuntimeError(f'Fitting is forbidden in publication verification: {frame.f_code.co_filename}')

    sys.setprofile(guard)
    threading.setprofile(guard)
    try:
        yield
    finally:
        sys.setprofile(previous)
        threading.setprofile(thread_previous)


def capture_hashes(output):
    destination = output / 'protected_hashes_before.csv'
    if destination.exists():
        raise FileExistsError(f'Baseline already exists: {destination}')
    paths = [p for folder in ('data', 'models', 'outputs')
             for p in (ROOT / folder).rglob('*') if p.is_file()]
    paths.extend([ROOT / 'environment.yml', *ROOT.glob('requirements*.txt')])
    presentation = {'data/category_codebook.csv', 'data/checksums.json'}
    rows = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': file_hash(p),
             'protected': p.relative_to(ROOT).as_posix() not in presentation}
            for p in sorted(paths)]
    save_csv(pd.DataFrame(rows), destination)


def verify_hashes(output):
    before = read_csv(output / 'protected_hashes_before.csv')
    after = before.copy()
    after['sha256'] = [file_hash(ROOT / path) for path in before.path]
    after['unchanged'] = after.sha256.eq(before.sha256)
    save_csv(after, output / 'protected_hashes_after.csv')
    changed = after.loc[after.protected & ~after.unchanged, 'path'].tolist()
    if changed:
        raise AssertionError(f'Protected files changed: {changed}')
    return int(after.protected.sum())


def round_trip(frame):
    decoded = decode_public_input(encode_public_input(frame), PUBLIC_VERSION)
    recovered = decoded[frame.columns]
    pd.testing.assert_frame_equal(recovered, frame, check_exact=True)
    return recovered


def verify_round_trips(output):
    rows = []
    for target in ('tnf', 'il10'):
        frames = [('all_endpoint_rows', load_data(target)), ('C4_cases', load_cases(target))]
        for seed, pipeline in load_models(target):
            before = joblib.hash(pipeline)
            preprocessor = pipeline.named_steps['preprocess']
            encoder = preprocessor.named_transformers_['categorical'].named_steps['onehot']
            categories = [values.copy() for values in encoder.categories_]
            names = preprocessor.get_feature_names_out().copy()
            for scope, frame in frames:
                recovered = round_trip(frame)
                original = preprocessor.transform(frame[feature_names()])
                restored = preprocessor.transform(recovered[feature_names()])
                np.testing.assert_array_equal(original, restored)
                direct = pipeline.predict(frame[feature_names()])
                coded = pipeline.predict(recovered[feature_names()])
                np.testing.assert_array_equal(direct, coded)
                rows.append({'target': target, 'seed': seed, 'scope': scope,
                             'rows': len(frame), 'transformed_columns': original.shape[1],
                             'max_transformed_difference': float(np.max(np.abs(original - restored))),
                             'max_prediction_difference_pg_ml': float(np.max(np.abs(direct - coded)))})
            assert before == joblib.hash(pipeline), (target, seed, 'pipeline state changed')
            np.testing.assert_array_equal(names, preprocessor.get_feature_names_out())
            for old, new in zip(categories, encoder.categories_, strict=True):
                np.testing.assert_array_equal(old, new)
    save_csv(pd.DataFrame(rows), output / 'model_round_trip.csv')
    return rows


def verify_domain_round_trips(output):
    rows = []
    for setting in read_json(DATA / 'APPLICABILITY_DOMAIN_SETTINGS.json'):
        target = setting['target']
        preprocessor = joblib.load(ROOT / f'models/{target}_DIAGNOSTIC_AD_PREPROCESSOR.joblib')
        state = joblib.hash(preprocessor)
        frame = load_data(target)
        recovered = round_trip(frame)
        original = preprocessor.transform(frame[FEATURES])
        restored = preprocessor.transform(recovered[FEATURES])
        np.testing.assert_array_equal(original, restored)
        training_array = preprocessor.transform(training(target)[FEATURES])
        direct = np.sort(pairwise_distances(original, training_array), axis=1)[:, :5].mean(axis=1)
        coded = np.sort(pairwise_distances(restored, training_array), axis=1)[:, :5].mean(axis=1)
        direct /= setting['threshold_q95']
        coded /= setting['threshold_q95']
        np.testing.assert_array_equal(direct, coded)
        assert state == joblib.hash(preprocessor)
        rows.append({'target': target, 'rows': len(frame),
                     'max_transformed_difference': float(np.max(np.abs(original - restored))),
                     'max_AD_ratio_difference': float(np.max(np.abs(direct - coded)))})
    save_csv(pd.DataFrame(rows), output / 'domain_round_trip.csv')
    return rows


def verify_original_outputs(workspace, output):
    isolated = workspace / 'verify'
    for folder in ('data/reference', 'data/validation_evidence', 'outputs'):
        shutil.copytree(ROOT / folder, isolated / folder, dirs_exist_ok=True)
    # Only the verifier's outputs are redirected; dataset and model loaders stay frozen.
    with patch.object(verify, 'ROOT', isolated), patch.object(predict, 'ROOT', isolated):
        verify.verify_results()
    predictions = csv(isolated / 'outputs/predictions/by_seed.csv')
    rows = []
    with np.load(ROOT / 'data/reference/accepted_predictions.npz') as reference:
        for target in ('tnf', 'il10'):
            oof = csv(ROOT / f'outputs/validation/{target}_OOF_predictions.csv')
            for seed in SEEDS[target]:
                current = predictions.loc[predictions.target.eq(target) & predictions.seed.eq(seed)].set_index('dataset_row_id')
                comparisons = [
                    ('OOF', oof.loc[oof.seed.eq(seed), 'prediction_pg_ml'].to_numpy()),
                    ('Test', current.loc[split_definition(target)['Test_IDs'], 'prediction_pg_ml'].to_numpy()),
                    ('C4', current.loc[['C4_Group2', 'C4_Group4'], 'prediction_pg_ml'].to_numpy()),
                ]
                for scope, actual in comparisons:
                    expected = reference[f'{target}_{seed}_{scope}']
                    np.testing.assert_array_equal(actual, expected)
                    rows.append({'target': target, 'seed': seed, 'scope': scope,
                                 'max_absolute_difference_pg_ml': float(np.max(np.abs(actual - expected)))})
    save_csv(pd.DataFrame(rows), output / 'frozen_reference_predictions.csv')
    return rows


def prepare_reference_replay(workspace):
    replay = workspace / 'pareto'
    for folder in ('search', 'results', 'checks'):
        (replay / folder).mkdir(parents=True, exist_ok=True)
    for path in (DATA / 'reference').iterdir():
        destination = replay / 'results' / path.stem if path.suffix == '.gz' else replay / 'results' / path.name
        if path.suffix == '.gz':
            with gzip.open(path, 'rb') as source, destination.open('wb') as output:
                shutil.copyfileobj(source, output)
        else:
            shutil.copy2(path, destination)
    values = csv(replay / 'results/ALL_CANDIDATES_AND_PREDICTIONS.csv')
    table(replay / 'search/CANDIDATE_INPUTS.csv',
          values[['candidate_id'] + FEATURES + ['material_tuple_id', 'experimental_tuple_id', 'profile_sha256']])
    shutil.copy2(DATA / 'reference/SEARCH_CONFIGURATION.json', replay / 'search/SEARCH_CONFIGURATION.json')
    learned, _ = verify_models(replay)
    levels = []
    for field in CAT:
        for value in SCHEMA['allowed_categories'][field]:
            included = value.lower() not in ('unclear', 'missing', 'unreported', 'unknown', '__missing__', '**missing**')
            included = included and all(training(target)[field].eq(value).any() and value in learned[target][field] for target in SEEDS)
            levels.append({'descriptor': field, 'value': value, 'included': included})
    table(replay / 'search/CATEGORICAL_LEVEL_SUPPORT.csv', levels)
    shap = csv(DATA / 'reference/REPRESENTATIVE_SHAP_ALL_REALISATIONS.csv')
    residuals = []
    for (target, candidate, realisation), group in shap.groupby(['target', 'candidate_id', 'model_realisation']):
        residuals.append({'target': target, 'candidate_id': candidate, 'model_realisation': realisation,
                          'additivity_residual_pg_ml': group.expected_value_pg_ml.iloc[0] + group.grouped_SHAP_pg_ml.sum() - group.prediction_pg_ml.iloc[0]})
    table(replay / 'checks/SHAP_ADDITIVITY.csv', residuals)
    return replay


def run(output, workspace):
    check_environment()
    check_inputs()
    verify_hashes(output)
    with forbid_fitting(), threadpool_limits(limits=1):
        model_rows = verify_round_trips(output)
        domain_rows = verify_domain_round_trips(output)
        original = verify_original_outputs(workspace, output)
        replay = prepare_reference_replay(workspace)
        verification.run(replay)
        cases = replay_cases(replay / 'checks')
    report = read_json(replay / 'checks/NUMERICAL_REPRODUCTION.json')
    save_csv(pd.DataFrame(cases), output / 'case_and_test_replay.csv')
    rows = [
        {'check': 'frozen_primary_predictions_round_trip', 'maximum_absolute_difference': max(r['max_prediction_difference_pg_ml'] for r in model_rows), 'scope': 'fresh predictions from each of six original pipelines, endpoint rows and C4'},
        {'check': 'frozen_primary_transforms_round_trip', 'maximum_absolute_difference': max(r['max_transformed_difference'] for r in model_rows), 'scope': 'exact arrays and unchanged learned state'},
        {'check': 'frozen_AD_round_trip', 'maximum_absolute_difference': max(r['max_AD_ratio_difference'] for r in domain_rows), 'scope': 'fresh saved diagnostic transforms and distances'},
        {'check': 'stored_validation_and_primary_Test_C4', 'maximum_absolute_difference': max(r['max_absolute_difference_pg_ml'] for r in original), 'scope': 'existing src.verify plus explicit OOF/Test/C4 deltas for six pipelines in isolated output directory'},
        {'check': 'separate_P15_and_primary_Test_C4', 'maximum_absolute_difference': max(r['max_absolute_difference_pg_ml'] for r in cases), 'scope': 'existing src.case_replay with original objects and reference tolerances'},
    ]
    rows.extend({'check': 'stored_Pareto_' + name, 'maximum_absolute_difference': difference,
                 'scope': 'existing reference tables and membership rules; no candidate generation or new SHAP calculation'}
                for name, difference in report['max_absolute_differences'].items())
    rows.append({'check': 'stored_SHAP_additivity_residual', 'maximum_absolute_difference': report['max_SHAP_additivity_residual_pg_ml'], 'scope': 'residual from unchanged stored SHAP table; tolerance 1e-7'})
    save_csv(pd.DataFrame(rows), output / 'numerical_checks.csv')
    count = verify_hashes(output)
    print(f'Verified {count} protected hashes; six primary models, separate P15, stored Pareto/SHAP, and exact publication round trips. Replay directory: {replay.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/publication_codes_20261019')
    parser.add_argument('--workspace', type=Path, default=ROOT / 'tmp/publication_codes_20261019')
    parser.add_argument('--capture', action='store_true')
    args = parser.parse_args()
    if args.capture:
        capture_hashes(args.output)
    else:
        run(args.output, args.workspace)


if __name__ == '__main__':
    main()
