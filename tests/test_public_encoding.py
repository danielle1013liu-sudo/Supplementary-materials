import csv

import numpy as np
import pandas as pd
import pytest

from src.data import ROOT, feature_names, load_cases, load_data, read_csv, read_json
from src.models import load_models
from src.hydrogel_immune_ml.public_encoding import (
    LEGACY_PATH, LEGACY_VERSION, PUBLIC_VERSION, VERSION_COLUMN,
    decode_public_input, encode_public_input, fields,
)


EXPECTED = {
    'polymer_origin_class': (['natural_polymer', 'hybrid_natural_synthetic', 'synthetic_polymer'], [168, 82, 22]),
    'material_bioactive_cue_class': (['none', 'anti_inflammatory_material_cue', 'anti_inflammatory_plus_growth_factor_material_cues', 'growth_factor_or_neuropeptide_material_cue', 'combined_pro_and_anti_inflammatory_material_cues', 'pro_inflammatory_material_cue'], [124, 81, 44, 10, 9, 4]),
    'gelation_or_crosslinking_mechanism': (['photo_crosslinking', 'physical_self_assembly', 'enzymatic_crosslinking', 'other'], [145, 34, 18, 75]),
    'network_density_proxy': (['medium', 'high', 'low'], [129, 122, 21]),
    'macrophage_model_class': (['RAW264_7', 'murine_primary_or_in_vivo_context', 'THP_1', 'human_monocyte_derived_macrophage', 'BMDM'], [185, 64, 16, 6, 1]),
    'stimulation_context': (['LPS', 'unstimulated_or_basal', 'IL4_IL13', 'LPS_IFNg', 'other_stimulation', 'unclear'], [121, 48, 14, 11, 51, 27]),
    'cell_material_interaction_mode': (['direct_surface_contact', 'hydrogel_extract_or_eluate', 'encapsulated_3D', 'in_vivo_tissue_or_systemic', 'transwell_or_indirect', 'unclear'], [144, 62, 29, 15, 8, 14]),
}


def test_shared_complete_bijective_mapping_and_frequency_order():
    schema = read_json(ROOT / 'data/feature_schema.json')
    pooled = pd.concat([load_data('tnf'), load_data('il10')], ignore_index=True)
    assert [f['machine_column'] for f in fields()] == list(EXPECTED)
    for field in fields():
        name, categories = field['machine_column'], field['categories']
        values, counts = EXPECTED[name]
        assert [c['machine_value'] for c in categories] == values
        assert set(values) == set(schema['allowed_categories'][name])
        assert [c['new_code'] for c in categories] == list('ABCDEF'[:len(values)])
        assert len({c['legacy_code'] for c in categories}) == len(values)
        assert [c['pooled_n'] for c in categories] == counts
        assert [int(pooled[name].eq(v).sum()) for v in values] == counts
        ordinary = [c['pooled_n'] for c in categories if c['machine_value'] not in field['residual_categories']]
        assert ordinary == sorted(ordinary, reverse=True)


def test_codebook_preserves_category_definitions_and_letter_mapping():
    legacy = read_csv(ROOT / LEGACY_PATH)
    revised = read_csv(ROOT / 'data/category_codebook.csv')
    keys = ['machine_column', 'machine_value']
    pd.testing.assert_frame_equal(
        legacy.drop(columns=['field_specific_code'] + [c for c in legacy if c.endswith('_n')]).set_index(keys).sort_index(),
        revised.drop(columns=['field_specific_code'] + [c for c in revised if c.endswith('_n')]).set_index(keys).sort_index(),
    )
    changes = read_csv(ROOT / 'data/public_codebooks/category_code_changes_20261019.csv')
    expected_rows = [(f['machine_column'], c['machine_value'], c['new_code']) for f in fields() for c in f['categories']]
    assert list(revised[['machine_column', 'machine_value', 'field_specific_code']].itertuples(index=False, name=None)) == expected_rows
    assert list(changes[['machine_column', 'machine_value', 'new_code']].itertuples(index=False, name=None)) == expected_rows
    for change in changes.itertuples():
        assert change.old_version == LEGACY_VERSION and change.new_version == PUBLIC_VERSION
        row = legacy.loc[legacy.machine_column.eq(change.machine_column) & legacy.machine_value.eq(change.machine_value)]
        assert row.field_specific_code.item() == change.old_code
    for field in fields():
        for category in field['categories']:
            row = legacy.loc[legacy.machine_column.eq(field['machine_column']) & legacy.machine_value.eq(category['machine_value'])]
            assert row.field_specific_code.item() == category['legacy_code']


@pytest.mark.parametrize('target', ['tnf', 'il10'])
@pytest.mark.parametrize('version', [LEGACY_VERSION, PUBLIC_VERSION])
def test_round_trip_preserves_identity_order_missingness_and_values(target, version):
    original = load_data(target)
    restored = decode_public_input(encode_public_input(original, version), version)
    pd.testing.assert_frame_equal(restored, original, check_exact=True)
    pd.testing.assert_frame_equal(restored.isna(), original.isna())


def test_all_categories_round_trip_even_if_absent_from_a_target():
    example = load_data('tnf').iloc[:1]
    for field in fields():
        for category in field['categories']:
            row = example.copy()
            row[field['machine_column']] = category['machine_value']
            for version in [LEGACY_VERSION, PUBLIC_VERSION]:
                pd.testing.assert_frame_equal(decode_public_input(encode_public_input(row, version), version), row)


def test_explicit_versions_and_no_double_migration():
    original = load_data('tnf').iloc[:2]
    coded = encode_public_input(original)
    with pytest.raises(ValueError, match='explicit'):
        decode_public_input(coded)
    with pytest.raises(ValueError, match='version does not match'):
        decode_public_input(coded, LEGACY_VERSION)
    with pytest.raises(ValueError, match='already versioned'):
        encode_public_input(coded)
    with pytest.raises(ValueError, match='Invalid machine_value'):
        encode_public_input(coded.drop(columns=VERSION_COLUMN))
    with pytest.raises(ValueError, match='Invalid new_code'):
        decode_public_input(original, PUBLIC_VERSION)
    with pytest.raises(ValueError, match='Specify'):
        decode_public_input(coded, 'frequency_codes_v1')
    coded.loc[coded.index[0], 'network_density_proxy'] = 'Z'
    with pytest.raises(ValueError, match='network_density_proxy'):
        decode_public_input(coded, PUBLIC_VERSION)


def test_same_letter_is_feature_and_version_specific():
    original = load_data('tnf').iloc[:1]
    legacy = encode_public_input(original, LEGACY_VERSION).drop(columns=VERSION_COLUMN)
    legacy['network_density_proxy'] = 'A'
    assert decode_public_input(legacy, LEGACY_VERSION).network_density_proxy.item() == 'low'
    assert decode_public_input(legacy, PUBLIC_VERSION).network_density_proxy.item() == 'medium'
    assert decode_public_input(original).equals(original)


def test_none_true_missing_and_historical_sentinels_remain_distinct():
    frame = load_data('tnf').iloc[:7].copy()
    frame['material_bioactive_cue_class'] = ['none', np.nan, None, pd.NA, '', '__missing__', '**missing**']
    for version in [LEGACY_VERSION, PUBLIC_VERSION]:
        encoded = encode_public_input(frame, version)
        restored = decode_public_input(encoded, version)
        pd.testing.assert_frame_equal(restored, frame, check_exact=True)
        assert encoded.material_bioactive_cue_class.iloc[0] == 'A'
        assert restored.material_bioactive_cue_class.iloc[-2:].tolist() == ['__missing__', '**missing**']


@pytest.mark.parametrize('target', ['tnf', 'il10'])
def test_publication_csv_preserves_canonical_numeric_text(target):
    with (ROOT / f'data/{target}.csv').open(encoding='utf-8-sig', newline='') as stream:
        original = list(csv.DictReader(stream))
    path = ROOT / f'data/publication/{target}_model_codes_20261019.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        published = list(csv.DictReader(stream))
    categorical = {f['machine_column'] for f in fields()}
    assert len(original) == len(published)
    for old, new in zip(original, published, strict=True):
        assert new[VERSION_COLUMN] == PUBLIC_VERSION
        assert {k: v for k, v in old.items() if k not in categorical} == {
            k: new[k] for k in old if k not in categorical
        }
    pd.testing.assert_frame_equal(decode_public_input(read_csv(path), PUBLIC_VERSION), load_data(target), check_exact=True)


@pytest.mark.parametrize('target', ['tnf', 'il10'])
def test_each_saved_preprocessor_and_model_exact_after_round_trip(target):
    original = pd.concat([load_data(target)[feature_names()], load_cases(target)[feature_names()]], ignore_index=True)
    restored = decode_public_input(encode_public_input(original), PUBLIC_VERSION)
    for _, model in load_models(target):
        preprocessor = model.named_steps['preprocess']
        encoder = preprocessor.named_transformers_['categorical'].named_steps['onehot']
        categories = [a.copy() for a in encoder.categories_]
        np.testing.assert_array_equal(preprocessor.transform(original), preprocessor.transform(restored))
        np.testing.assert_array_equal(model.predict(original), model.predict(restored))
        for before, after in zip(categories, encoder.categories_, strict=True):
            np.testing.assert_array_equal(before, after)
