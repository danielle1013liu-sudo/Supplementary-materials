"""Predict from canonical or explicitly versioned publication CSV inputs."""

import argparse
import os
from pathlib import Path
import pandas as pd

from .data import ROOT, ID, check_environment, check_inputs, feature_names, read_csv, save_csv
from .hydrogel_immune_ml.public_encoding import decode_public_input
from .models import load_models


def validate_output_path(input_path, output_path):
    """Keep predictions separate from their inputs and distributed repository files."""
    source, destination = Path(input_path).resolve(), Path(output_path).resolve()
    if destination == source or (
        source.exists() and destination.exists() and destination.samefile(source)
    ):
        raise ValueError('Prediction output must not overwrite or alias the input file')
    protected = ('data', 'models', 'src', 'docs', 'tests', '.git', 'configs')
    if destination == ROOT or any(destination.is_relative_to(ROOT / name) for name in protected):
        raise ValueError('Prediction output is inside a protected repository path')
    if destination.exists() and destination.is_relative_to(ROOT) and not destination.is_relative_to(ROOT / 'tmp'):
        raise ValueError('Prediction output must not overwrite an existing repository file; choose a new path or tmp/')
    # resolve() follows symlinks; hard links need a file-identity check.
    if destination.exists() and destination.stat().st_nlink > 1:
        for directory, folders, files in os.walk(ROOT):
            folders[:] = [name for name in folders if Path(directory) / name != ROOT / 'tmp']
            if any(destination.samefile(Path(directory) / name) for name in files):
                raise ValueError('Prediction output must not alias a protected repository file')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--target', required=True, choices=['tnf', 'il10'])
    parser.add_argument('--codebook-version', default='canonical')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = validate_output_path(args.input, args.output)
    check_environment()
    check_inputs()
    frame = decode_public_input(read_csv(args.input), args.codebook_version)
    if ID not in frame or frame[ID].isna().any() or not frame[ID].is_unique:
        raise ValueError(f'Input requires unique, non-missing {ID} values')
    records = []
    for seed, pipeline in load_models(args.target):
        records.append(pd.DataFrame({
            ID: frame[ID], 'target': args.target, 'seed': seed,
            'prediction_pg_ml': pipeline.predict(frame[feature_names()]),
        }))
    save_csv(pd.concat(records, ignore_index=True), output)


if __name__ == '__main__':
    main()
