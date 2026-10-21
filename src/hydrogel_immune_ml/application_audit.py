"""Check downloaded browser predictions against the unchanged released pipelines."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ..data import ROOT, file_hash, read_json
from ..models import load_models
from .inference import form_frame
from .application_schema import validate_application_input
from .public_encoding import decode_public_input


def main(output=None, inputs=None):
    output = Path(output) if output else ROOT / 'results/revision'
    if inputs is not None:
        inputs = Path(inputs)
        frame = pd.read_csv(inputs, float_precision='round_trip', keep_default_na=False, na_values=[''])
        frame = validate_application_input(decode_public_input(frame, 'canonical'))
        if len(frame) != 1:
            raise ValueError('Expected one downloaded browser input profile')
    else:
        values = read_json(ROOT / 'results/revision/browser_case.json')
        frame = form_frame({'values': values, 'unreported': {}})
    downloaded = pd.read_csv(output / 'browser_predictions.csv', float_precision='round_trip')
    checks = []
    for target in ('tnf', 'il10'):
        originals = []
        for seed, pipeline in load_models(target):
            original = float(pipeline.predict(frame)[0])
            originals.append(original)
            actual = downloaded.loc[downloaded.target.eq(target) & downloaded.realisation.eq(str(seed)), 'prediction_pg_ml'].item()
            checks.append({'target': target, 'realisation': str(seed), 'quantity': 'prediction_pg_ml', 'original': original, 'browser_download': actual, 'absolute_difference': abs(original - actual)})
        mean_row = downloaded.loc[downloaded.target.eq(target) & downloaded.realisation.eq('mean')].iloc[0]
        for name, expected in [('prediction_pg_ml', np.mean(originals)), ('sample_SD_pg_ml', np.std(originals, ddof=1))]:
            checks.append({'target': target, 'realisation': 'mean', 'quantity': name, 'original': expected, 'browser_download': mean_row[name], 'absolute_difference': abs(expected - mean_row[name])})
    table = pd.DataFrame(checks)
    if inputs is not None:
        table['input_source'] = str(inputs)
        table['input_sha256'] = file_hash(inputs)
    table.to_csv(output / 'browser_backend_equality.csv', index=False, float_format='%.17g')
    assert all(row['absolute_difference'] == 0 for row in checks)
    print('Browser downloads equal the six original predictions, means and sample SDs exactly.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--inputs', type=Path)
    args = parser.parse_args()
    main(args.output, args.inputs)
