"""Cloud runtime checks without changing the frozen reproduction environment."""

from datetime import datetime, timezone
import importlib.metadata
import platform

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..data import ROOT, feature_names, load_cases, read_json, split_data


def check_cloud_environment(output=None):
    output = ROOT / 'results/runtime' if output is None else output
    output.mkdir(parents=True, exist_ok=True)
    versions = read_json(ROOT / 'data/software_versions.json')
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = []
    for name, expected in versions.items():
        if name == 'python':
            actual = platform.python_version()
            passed = actual.split('.')[:2] == expected.split('.')[:2]
            requirement = '.'.join(expected.split('.')[:2]) + '.*'
        else:
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                actual = 'not installed'
            passed = actual == expected
            requirement = expected
        rows.append({'checked_at_utc': timestamp, 'runtime_profile': 'community-cloud',
                     'package': name, 'frozen_version': expected,
                     'required_version': requirement, 'actual_version': actual,
                     'passed': passed})
    table = pd.DataFrame(rows)
    table.to_csv(output / 'environment.csv', index=False)
    rejected = table.loc[~table.passed, 'package'].tolist()
    if rejected:
        raise ValueError('Unsupported application runtime: ' + ', '.join(rejected))


def verify_cloud_predictions(resources, output=None):
    output = ROOT / 'results/runtime' if output is None else output
    output.mkdir(parents=True, exist_ok=True)
    checks, values = [], []
    with np.load(ROOT / 'data/reference/accepted_predictions.npz') as frozen:
        with threadpool_limits(limits=1):
            for target, resource in resources.items():
                training, test = split_data(target)
                scopes = [('Training', training), ('Test', test), ('C4', load_cases(target))]
                for seed, pipeline in resource['models']:
                    for scope, frame in scopes:
                        actual = np.asarray(pipeline.predict(frame[feature_names()]))
                        expected = frozen[f'{target}_{seed}_{scope}']
                        same_shape = actual.shape == expected.shape
                        passed = same_shape and np.isfinite(actual).all() and np.array_equal(actual, expected)
                        checks.append({'target': target, 'seed': seed, 'scope': scope,
                                       'actual_shape': str(actual.shape),
                                       'reference_shape': str(expected.shape), 'passed': bool(passed)})
                        if same_shape:
                            identities = frame['dataset_row_id'] if 'dataset_row_id' in frame else frame['hydrogel_group']
                            values.extend({'target': target, 'seed': seed, 'scope': scope,
                                           'row_identity': identity, 'actual_pg_ml': value,
                                           'release_pg_ml': reference, 'difference_pg_ml': value - reference}
                                          for identity, value, reference in zip(identities, actual, expected, strict=True))
    pd.DataFrame(values).to_csv(output / 'prediction_replay.csv', index=False, float_format='%.17g')
    summary = pd.DataFrame(checks)
    summary.to_csv(output / 'prediction_replay_checks.csv', index=False)
    if summary.empty or not summary.passed.all():
        raise ValueError('Cloud runtime predictions differ from the frozen release; prediction is disabled.')
    return summary
