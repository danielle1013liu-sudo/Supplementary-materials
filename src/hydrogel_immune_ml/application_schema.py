"""Application input checks and identities of fitted encoded columns."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from ..data import ROOT, feature_names, read_json
from .public_encoding import fields


def validate_application_input(frame):
    missing = set(feature_names()) - set(frame)
    if missing:
        raise ValueError(f'Missing descriptor columns: {sorted(missing)}')
    for field in fields():
        name = field['machine_column']
        if frame[name].isna().any() or frame[name].eq('').any():
            raise ValueError(f'Choose {field["feature"]}; a blank choice is not Unclear')
        allowed = {c['machine_value'] for c in field['categories']}
        unknown = set(frame[name]) - allowed
        if unknown:
            raise ValueError(f'Unknown category for {field["feature"]}: {sorted(unknown)}')
    result = frame[feature_names()].copy()
    numeric = read_json(ROOT / 'data/feature_schema.json')['numeric_semantics']
    for name, details in numeric.items():
        try:
            result[name] = pd.to_numeric(result[name], errors='raise')
        except (ValueError, TypeError) as error:
            raise ValueError(f'{details["display_name"]} must be a number or explicitly unreported') from error
        reported = result[name].dropna().to_numpy(float)
        if not np.isfinite(reported).all() or (reported < 0).any():
            raise ValueError(f'{details["display_name"]} must be finite and non-negative')
    return result


def fitted_columns(preprocessor):
    """Read identities from fitted metadata without splitting category strings."""
    identities = []
    for name, transformer, columns in preprocessor.transformers_:
        if isinstance(transformer, str) and transformer == 'drop':
            continue
        output = preprocessor.output_indices_[name]
        if output.stop == output.start:
            continue
        steps = getattr(transformer, 'steps', [('transform', transformer)])
        encoders = [step for _, step in steps if isinstance(step, OneHotEncoder)]
        if encoders:
            encoder = encoders[0]
            if encoder.drop is not None or encoder.min_frequency is not None or encoder.max_categories is not None:
                raise ValueError('Unexpected grouped or dropped fitted encoder')
            local = [(column, category) for column, categories in
                     zip(columns, encoder.categories_, strict=True) for category in categories]
        else:
            local = [(column, None) for column in columns]
        if len(local) != output.stop - output.start or len(identities) != output.start:
            raise ValueError(f'Unexpected fitted output width/order: {name}')
        identities.extend(local)
    return identities


def unsupported_categories(frame, identities):
    supported = {}
    for name, category in identities:
        if category is not None:
            supported.setdefault(name, set()).add(category)
    return {name: sorted(set(frame[name].dropna()) - categories)
            for name, categories in supported.items()
            if set(frame[name].dropna()) - categories}
