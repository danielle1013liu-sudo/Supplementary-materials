"""Feature-specific publication labels; model inputs remain machine values."""

import pandas as pd

from ..data import ROOT, configuration, feature_names, read_json

PUBLIC_VERSION = 'frequency_codes_20261019'
LEGACY_VERSION = 'legacy_codes_v1'
VERSION_COLUMN = 'publication_codebook_version'
LEGACY_PATH = 'data/public_codebooks/legacy_codes_v1.csv'


def codebook():
    return read_json(ROOT / f'data/public_codebooks/{PUBLIC_VERSION}.json')


def fields():
    return codebook()['fields']


def _frame(values):
    frame = pd.DataFrame([values]) if isinstance(values, dict) else values.copy()
    missing = set(feature_names()) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing descriptor columns: {sorted(missing)}')
    return frame


def _code_key(version):
    if version not in (PUBLIC_VERSION, LEGACY_VERSION):
        raise ValueError(f'Specify {PUBLIC_VERSION} or {LEGACY_VERSION} for letter-coded input')
    return 'new_code' if version == PUBLIC_VERSION else 'legacy_code'


def _translate(frame, source, destination):
    sentinels = {v['categorical_imputer']['fill_value'] for v in configuration()['targets'].values()}
    for field in fields():
        name = field['machine_column']
        mapping = {c[source]: c[destination] for c in field['categories']}
        # Historical missing tokens are not the valid cue category 'none'.
        untouched = frame[name].isna() | frame[name].isin(sentinels | {''})
        unknown = ~untouched & ~frame[name].isin(mapping)
        if unknown.any():
            raise ValueError(f'Invalid {source} for {name}: {frame.loc[unknown, name].tolist()}')
        frame.loc[~untouched, name] = frame.loc[~untouched, name].map(mapping)
    return frame


def encode_public_input(values, version=PUBLIC_VERSION):
    """Encode canonical values while preserving rows, numeric cells and missingness."""
    key = _code_key(version)
    frame = _frame(values)
    if VERSION_COLUMN in frame:
        raise ValueError('Input is already versioned; decode it before encoding again')
    frame = _translate(frame, 'machine_value', key)
    frame[VERSION_COLUMN] = version
    return frame


def decode_public_input(values, codebook_version='canonical'):
    """Decode explicitly versioned letters before passing descriptors to a pipeline."""
    frame = _frame(values)
    if codebook_version == 'canonical':
        if VERSION_COLUMN in frame:
            raise ValueError('Letter-coded input requires an explicit matching codebook version')
        return _translate(frame, 'machine_value', 'machine_value')
    key = _code_key(codebook_version)
    if VERSION_COLUMN in frame:
        if not frame[VERSION_COLUMN].eq(codebook_version).all():
            raise ValueError('Input codebook version does not match the requested version')
        frame = frame.drop(columns=VERSION_COLUMN)
    return _translate(frame, key, 'machine_value')
