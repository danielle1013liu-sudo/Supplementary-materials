"""Export publication codes without rounding the canonical numeric text."""

import argparse
import csv
from pathlib import Path

import pandas as pd

from ..data import ROOT, load_data, read_csv, save_csv, split_data
from .public_encoding import PUBLIC_VERSION, encode_public_input, fields


def verify_counts(output):
    tables = {target: load_data(target) for target in ('tnf', 'il10')}
    splits = {target: split_data(target) for target in tables}
    book = read_csv(ROOT / 'data/category_codebook.csv').set_index(['machine_column', 'machine_value'])
    records, partitions = [], []
    for field in fields():
        name = field['machine_column']
        for category in field['categories']:
            value = category['machine_value']
            counts = {t: int(df[name].eq(value).sum()) for t, df in tables.items()}
            row = book.loc[(name, value)]
            for target, count in counts.items():
                if count != category[f'{target}_n'] or count != row[f'{target}_total_n']:
                    raise ValueError(f'Count discrepancy for {name}/{value}/{target}: {count}')
                for subset, frame in zip(('development', 'test'), splits[target], strict=True):
                    ids = frame.loc[frame[name].eq(value), 'dataset_row_id'].tolist()
                    recorded = int(row[f'{target}_{subset}_n'])
                    # Retain pre-existing partition counts without revising the frozen split.
                    partitions.append({'machine_column': name, 'machine_value': value,
                                       'target': target, 'subset': subset, 'recorded_n': recorded,
                                       'frozen_split_n': len(ids), 'equal': recorded == len(ids),
                                       'current_row_ids': '|'.join(ids) if recorded != len(ids) else ''})
            if sum(counts.values()) != category['pooled_n']:
                raise ValueError(f'Pooled count discrepancy for {name}/{value}')
            records.append({'machine_column': name, 'code': category['new_code'],
                            'machine_value': value, 'tnf_n': counts['tnf'],
                            'il10_n': counts['il10'], 'pooled_n': sum(counts.values())})
    save_csv(pd.DataFrame(records), Path(output) / 'category_counts.csv')
    save_csv(pd.DataFrame(partitions), Path(output) / 'category_partition_count_audit.csv')


def export_canonical_codes(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for target in ('tnf', 'il10'):
        with (ROOT / f'data/{target}.csv').open(encoding='utf-8-sig', newline='') as stream:
            frame = pd.DataFrame(list(csv.DictReader(stream)))
        coded = encode_public_input(frame)
        coded.to_csv(output / f'{target}_model_codes_20261019.csv', index=False, lineterminator='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/publication')
    parser.add_argument('--checks', type=Path, default=ROOT / 'results/publication_codes_20261019')
    args = parser.parse_args()
    verify_counts(args.checks)
    export_canonical_codes(args.output)
    print(f'Exported canonical rows using {PUBLIC_VERSION}')


if __name__ == '__main__':
    main()
