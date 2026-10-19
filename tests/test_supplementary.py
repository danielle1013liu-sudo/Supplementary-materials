from collections import Counter
from zipfile import ZipFile

import pytest

from src.data import ROOT
from src.hydrogel_immune_ml.public_encoding import PUBLIC_VERSION, VERSION_COLUMN, fields
from src.hydrogel_immune_ml.supplementary import (
    CODE_COLUMNS, MEMBER, NOTE, export_tables, migrate, publication_rows,
    revised_xml, table_cells, text_positions, verify_semantics,
)

SOURCE = ROOT / 'docs/source/Supplementary_01Oct26_YL.docx'
REVISED = ROOT / 'docs/Supplementary_frequency_ordered_20261019.docx'


def xml(path):
    with ZipFile(path) as archive:
        return archive.read(MEMBER).decode('utf-8')


def test_full_document_semantics_and_unchanged_formatting():
    before, after = xml(SOURCE), xml(REVISED)
    assert after == revised_xml(before)
    assert after.count(NOTE) == 1
    assert verify_semantics(before, after)['categorical_cells_verified'] == 1904
    with ZipFile(SOURCE) as source, ZipFile(REVISED) as revised:
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
    assert first.read_bytes() == REVISED.read_bytes()
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
