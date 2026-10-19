"""Predict from canonical or explicitly versioned publication CSV inputs."""

import argparse
import pandas as pd

from .data import ID, check_environment, check_inputs, feature_names, read_csv, save_csv
from .hydrogel_immune_ml.public_encoding import decode_public_input
from .models import load_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--target', required=True, choices=['tnf', 'il10'])
    parser.add_argument('--codebook-version', default='canonical')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
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
    save_csv(pd.concat(records, ignore_index=True), args.output)


if __name__ == '__main__':
    main()
