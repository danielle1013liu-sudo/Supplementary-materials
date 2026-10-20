from collections import Counter
from zipfile import ZipFile

import pytest

from src.data import ROOT
from src.hydrogel_immune_ml.public_encoding import PUBLIC_VERSION, VERSION_COLUMN, fields
from src.hydrogel_immune_ml.supplementary import (
    CODE_COLUMNS, MEMBER, NOTE, export_tables, migrate, publication_rows,
    revised_xml, table_cells, text_positions, verify_semantics, verify_release_document,
    CURRENT_DOCUMENT, CORRECTED_IDS, ROW_MAP,
)

SOURCE = ROOT / 'docs/source/Supplementary_01Oct26_YL.docx'
FREQUENCY = ROOT / 'docs/source/Supplementary_frequency_ordered_20261019.docx'
REVISED = ROOT / CURRENT_DOCUMENT


def xml(path):
    with ZipFile(path) as archive:
        return archive.read(MEMBER).decode('utf-8')


def test_full_document_semantics_and_unchanged_formatting():
    before, after = xml(SOURCE), xml(FREQUENCY)
    assert after == revised_xml(before)
    assert after.count(NOTE) == 1
    assert verify_semantics(before, after)['categorical_cells_verified'] == 1904
    with ZipFile(SOURCE) as source, ZipFile(FREQUENCY) as revised:
        assert source.namelist() == revised.namelist()
        for name in source.namelist():
            if name != MEMBER:
                assert source.read(name) == revised.read(name)


def test_complete_reordered_legend_and_pooled_counts():
    before, after = table_cells(xml(SOURCE)), table_cells(xml(REVISED))
    for index, field in enumerate(fields()):
        original = dict(part.split(', ', 1) for part in before[1][index + 1][1][0].split('; '))
        legend = after[1][index + 1][1][0].split('; ')
        assert legend == [c['new_code'] + ', ' + original[c['legacy_code']] for c in field['categories']]
        counts = Counter(row[CODE_COLUMNS[index]][0] for ti in (15, 16) for row in after[ti][1:])
        assert counts == {c['new_code']: c['pooled_n'] for c in field['categories']}


def test_version_detection_is_idempotent_and_rejects_mixed(tmp_path):
    first, second = tmp_path / 'first.docx', tmp_path / 'second.docx'
    report = migrate(SOURCE, first, SOURCE)
    assert report['detected_input_version'] == 'legacy_codes_v1'
    assert first.read_bytes() == FREQUENCY.read_bytes()
    report = migrate(first, second, SOURCE)
    assert report['detected_input_version'] == PUBLIC_VERSION
    assert report['changes_applied_this_run'] == 0
    assert first.read_bytes() == second.read_bytes()
    mixed = tmp_path / 'mixed.docx'
    with ZipFile(first) as source, ZipFile(mixed, 'w') as destination:
        for member in source.infolist():
            data = source.read(member.filename)
            if member.filename == MEMBER:
                data = data.replace(b'A, medium; B, high; C, low', b'B, medium; B, high; C, low', 1)
            destination.writestr(member, data)
    with pytest.raises(ValueError, match='Mixed, ambiguous'):
        migrate(mixed, tmp_path / 'rejected.docx', SOURCE)


def test_word_exports_preserve_all_cells_and_record_identity(tmp_path):
    export_tables(REVISED, tmp_path)
    after = xml(REVISED)
    for target, count in (('tnf', 139), ('il10', 133)):
        filename = f'{target}_supplementary_codes_20261019.csv'
        assert (tmp_path / filename).read_bytes() == (ROOT / 'data/publication' / filename).read_bytes()
        records = publication_rows(after, target)
        assert len(records) == count
        assert len({row['dataset_row_id'] for row in records}) == count
        assert [row['supplementary_no'] for row in records] == [str(i) for i in range(1, count + 1)]
        assert {row[VERSION_COLUMN] for row in records} == {PUBLIC_VERSION}


def test_run_split_and_entity_offsets():
    text, offsets = text_positions('<w:r><w:t>A</w:t></w:r><w:r><w:t>, a &amp; b</w:t></w:r>')
    assert text == 'A, a & b'
    assert len(offsets) == len(text)


def test_release_sources_and_only_seven_dataset_corrections():
    import json
    record = verify_release_document(REVISED)
    assert set(record['corrected_IL10_ids']) == CORRECTED_IDS
    before, after = table_cells(xml(FREQUENCY)), table_cells(xml(REVISED))
    identities = json.loads((ROOT / ROW_MAP).read_text())
    changed = set()
    for target, table_index in [('tnf', 15), ('il10', 16)]:
        for ri, (old_row, new_row) in enumerate(zip(before[table_index], after[table_index], strict=True)):
            for ci, (old, new) in enumerate(zip(old_row, new_row, strict=True)):
                if old[0] != new[0]:
                    assert target == 'il10' and ci == 7 and ri > 0
                    changed.add(identities[target][ri - 1]['dataset_row_id'])
    assert changed == CORRECTED_IDS


def test_release_selected_model_and_current_case_values():
    import pandas as pd
    tables = table_cells(xml(REVISED))
    assert tables[8][2][2][0] == 'Median'
    assert tables[8][9][2][0] == '8'
    assert tables[7][2][3][0] == '0.778 ± 0.004'
    case = pd.read_csv(ROOT / 'outputs/case_studies/summary.csv')
    for row in tables[14][1:]:
        for target, column in [('tnf', 2), ('il10', 4)]:
            r = case.loc[(case.target == target) & (case.formulation == row[0][0])].iloc[0]
            assert row[column][0] == f'{r.prediction_mean_pg_ml:.3f} ± {r.prediction_sample_SD_pg_ml:.3f}'


def test_release_figures_are_current_uncropped_results():
    from docx import Document
    import hashlib
    document = Document(REVISED)
    expected = ['parity_predictions', 'tnf_shap_landscape', 'il10_shap_landscape', 'context_shap_landscapes']
    pictures = [p for p in document.paragraphs if p._p.xpath('.//a:blip')]
    assert len(pictures) == len(expected)
    for paragraph, name in zip(pictures, expected, strict=True):
        blip = paragraph._p.xpath('.//a:blip')[0]
        rid = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
        assert document.part.related_parts[rid].blob == (ROOT / f'outputs/figures/{name}.png').read_bytes()
        assert not paragraph._p.xpath('.//a:srcRect')


@pytest.fixture
def published_document_checkout(tmp_path):
    """Use actual published bytes, omitting the ignored local Pareto replay."""
    import hashlib
    import json
    import shutil
    from src.hydrogel_immune_ml.supplementary import RELEASE_MANIFEST

    record = json.loads((ROOT / RELEASE_MANIFEST).read_text(encoding='utf-8'))
    snapshot_path = 'data/pareto/CURRENT_RELEASE.json'
    snapshot = json.loads((ROOT / snapshot_path).read_text(encoding='utf-8'))
    published = {
        name.replace('outputs/pareto/results/', 'data/pareto/current/results/')
        if name.startswith('outputs/pareto/results/') else name
        for name in record['inputs_sha256']
    }
    for name in published | set(snapshot['source_sha256']) | {snapshot_path, CURRENT_DOCUMENT}:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    # A release-builder records the current source hash after an intentional code change.
    module = 'src/hydrogel_immune_ml/supplementary.py'
    record['inputs_sha256'][module] = hashlib.sha256((tmp_path / module).read_bytes()).hexdigest()
    (tmp_path / RELEASE_MANIFEST).write_text(json.dumps(record), encoding='utf-8')
    assert not (tmp_path / 'outputs/pareto').exists()
    return tmp_path


def test_release_verification_uses_published_snapshot_without_local_replay(published_document_checkout):
    root = published_document_checkout
    record = verify_release_document(root / CURRENT_DOCUMENT, root=root)
    assert record['pareto_profiles'] == 18


@pytest.mark.parametrize('changed', ['published_table', 'local_table', 'snapshot_digest', 'accepted_source'])
def test_published_document_fallback_preserves_hash_guards(published_document_checkout, changed):
    import json
    root = published_document_checkout
    table = 'FINAL_PARETO_PROFILES.csv'
    if changed == 'published_table':
        path = root / 'data/pareto/current/results' / table
        path.write_bytes(path.read_bytes() + b'changed')
    elif changed == 'local_table':
        # An existing but invalid local replay must not silently fall back.
        path = root / 'outputs/pareto/results' / table
        path.parent.mkdir(parents=True)
        path.write_bytes(b'changed')
    elif changed == 'snapshot_digest':
        path = root / 'data/pareto/CURRENT_RELEASE.json'
        record = json.loads(path.read_text(encoding='utf-8'))
        record['snapshot_sha256']['data/pareto/current/results/' + table] = '0' * 64
        path.write_text(json.dumps(record), encoding='utf-8')
    else:
        path = root / 'models/manifest.json'
        path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='changed|does not match|differs'):
        verify_release_document(root / CURRENT_DOCUMENT, root=root)
