"""Export publication codes without rounding the canonical numeric text."""

import argparse
import csv
from pathlib import Path

import pandas as pd

from ..data import ROOT, load_data, read_csv, save_csv, read_json, save_json, file_hash, split_definition
from .public_encoding import DATASET_VERSION, PUBLIC_VERSION, encode_public_input, fields


def verify_counts(output):
    tables = {target: load_data(target) for target in ('tnf', 'il10')}
    book = read_csv(ROOT / 'data/category_codebook.csv').set_index(['machine_column', 'machine_value'])
    records = []
    for field in fields():
        name = field['machine_column']
        for category in field['categories']:
            value = category['machine_value']
            counts = {t: int(df[name].eq(value).sum()) for t, df in tables.items()}
            row = book.loc[(name, value)]
            for target, count in counts.items():
                if count != category[f'{target}_n'] or count != row[f'{target}_total_n']:
                    raise ValueError(f'Count discrepancy for {name}/{value}/{target}: {count}')
            if sum(counts.values()) != category['pooled_n']:
                raise ValueError(f'Pooled count discrepancy for {name}/{value}')
            records.append({'machine_column': name, 'code': category['new_code'],
                            'machine_value': value, 'tnf_n': counts['tnf'],
                            'il10_n': counts['il10'], 'pooled_n': sum(counts.values())})
    save_csv(pd.DataFrame(records), Path(output) / 'category_counts.csv')


def export_canonical_codes(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for target in ('tnf', 'il10'):
        with (ROOT / f'data/{target}.csv').open(encoding='utf-8-sig', newline='') as stream:
            frame = pd.DataFrame(list(csv.DictReader(stream)))
        coded = encode_public_input(frame)
        coded.to_csv(output / f'{target}_model_codes_20261019.csv', index=False, lineterminator='\n')


def refresh_counts():
    """Refresh occurrence counts without reassigning any publication letter."""
    path = ROOT / f'data/public_codebooks/{PUBLIC_VERSION}.json'
    book = read_json(path)
    tables = {t: load_data(t) for t in ('tnf', 'il10')}
    counts = read_csv(ROOT / 'data/category_codebook.csv')
    for field in book['fields']:
        name = field['machine_column']
        for category in field['categories']:
            value = category['machine_value']
            selected = counts.machine_column.eq(name) & counts.machine_value.eq(value)
            if selected.sum() != 1:
                raise ValueError(f'Ambiguous codebook category: {name}/{value}')
            for target, table in tables.items():
                mask = table[name].eq(value)
                split = split_definition(target)
                category[f'{target}_n'] = int(mask.sum())
                for scope, ids in [('total', table.dataset_row_id),
                                   ('development', split['Training_IDs']), ('test', split['Test_IDs'])]:
                    counts.loc[selected, f'{target}_{scope}_n'] = int((mask & table.dataset_row_id.isin(ids)).sum())
            category['pooled_n'] = category['tnf_n'] + category['il10_n']
    book['dataset_version'] = DATASET_VERSION
    book['policy'] = ('The frequency_codes_20261019 letter mapping is fixed. Letters are nominal '
                      'identifiers, including crosslink level. Counts reflect the corrected 20261020 '
                      'dataset; categories are not reassigned when counts change. None and basal '
                      'conditions are valid categories.')
    save_json(book, path)
    save_csv(counts, ROOT / 'data/category_codebook.csv')


def export_manifest(output):
    output = Path(output)
    save_json({'dataset_version': DATASET_VERSION, 'publication_codebook_version': PUBLIC_VERSION,
               'canonical_sha256': {t: file_hash(ROOT / f'data/{t}.csv') for t in ('tnf', 'il10')},
               'mapping_sha256': file_hash(ROOT / f'data/public_codebooks/{PUBLIC_VERSION}.json'),
               'note': 'Filename dates identify the fixed letter mapping; dataset_version identifies the data revision.'},
              output / 'release_manifest.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/publication')
    parser.add_argument('--checks', type=Path, default=ROOT / 'results/publication_codes_20261020')
    args = parser.parse_args()
    refresh_counts()
    verify_counts(args.checks)
    export_canonical_codes(args.output)
    export_manifest(args.output)
    print(f'Exported canonical rows using {PUBLIC_VERSION}')


if __name__ == '__main__':
    main()
