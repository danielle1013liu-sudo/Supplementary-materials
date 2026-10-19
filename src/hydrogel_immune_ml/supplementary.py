"""Migrate the verified supplementary tables without rebuilding Word formatting."""

import argparse
import csv
from decimal import Decimal
import hashlib
import html
import json
from pathlib import Path
import re
from zipfile import ZipFile

from .public_encoding import PUBLIC_VERSION, VERSION_COLUMN, fields

ROOT = Path(__file__).resolve().parents[2]
SOURCE_HASH = '2ae231785c23986fb55c545656dd553463f0a708abe3c3633c095c89d207c189'
MEMBER = 'word/document.xml'
CODE_COLUMNS = [2, 5, 6, 7, 8, 9, 10]
NUMERIC_COLUMNS = {3: 'modulus_value_kPa', 4: 'pore_size_um',
                   11: 'culture_or_exposure_duration_h', 12: 'corrected_value_pg_ml'}
NOTE = ('Codes are nominal labels, ordered by pooled frequency in Tables S8-S9, '
        'with other/mixed and unclear categories placed last.')
CAPTION = 'Categorical codes used in the dataset tables.'


def elements(xml, tag):
    return list(re.finditer(r'<w:' + tag + r'\b[^>]*>.*?</w:' + tag + '>', xml, re.S))


def text_positions(xml):
    text, positions = [], []
    for node in re.finditer(r'<w:t(?:\s[^>]*)?>(.*?)</w:t>', xml, re.S):
        for token in re.finditer(r'&[^;]+;|.', node.group(1), re.S):
            value = html.unescape(token.group())
            text.extend(value)
            positions.extend([node.start(1) + token.start()] * len(value))
    return ''.join(text), positions


def table_cells(xml):
    tables = []
    for table in elements(xml, 'tbl'):
        rows = []
        for row in elements(table.group(), 'tr'):
            cells = []
            for cell in elements(row.group(), 'tc'):
                text, offsets = text_positions(cell.group())
                base = table.start() + row.start() + cell.start()
                cells.append((text, [base + offset for offset in offsets]))
            rows.append(cells)
        tables.append(rows)
    if len(tables) != 17 or [len(tables[i]) for i in (1, 15, 16)] != [8, 140, 134]:
        raise ValueError('Supplementary table structure differs from the verified source')
    return tables


def replace_spans(xml, changes):
    for start, end, old, new in sorted(changes, reverse=True):
        if xml[start:end] != old:
            raise ValueError('Target XML text differs from the planned replacement')
        xml = xml[:start] + new + xml[end:]
    return xml


def revised_xml(original):
    tables = table_cells(original)
    changes = []
    for index, field in enumerate(fields()):
        legend = tables[1][index + 1]
        if legend[0][0].casefold() != field['feature'].casefold():
            raise ValueError('Unexpected S1b descriptor order')
        text, offsets = legend[1]
        phrases = dict(part.split(', ', 1) for part in text.split('; '))
        categories = field['categories']
        if set(phrases) != {category['legacy_code'] for category in categories}:
            raise ValueError('S1b is not a complete legacy legend')
        new_text = '; '.join(category['new_code'] + ', ' + phrases[category['legacy_code']]
                             for category in categories)
        # The verified legend has one text run; keep its existing font and paragraph markup.
        old_xml = original[offsets[0]:offsets[-1] + 1]
        if html.unescape(old_xml) != text:
            raise ValueError('Unexpected run splitting in S1b')
        changes.append((offsets[0], offsets[-1] + 1, old_xml, html.escape(new_text, quote=False)))
        mapping = {category['legacy_code']: category['new_code'] for category in categories}
        for table_index in (15, 16):
            for row in tables[table_index][1:]:
                code, positions = row[CODE_COLUMNS[index]]
                if code not in mapping:
                    raise ValueError('Invalid source category code')
                changes.append((positions[0], positions[0] + 1, code, mapping[code]))
    if original.count(CAPTION) != 1:
        raise ValueError('Expected one S1b caption')
    start = original.index(CAPTION)
    changes.append((start, start + len(CAPTION), CAPTION, CAPTION + ' ' + NOTE))
    result = replace_spans(original, changes)
    if re.findall(r'<[^>]+>', original) != re.findall(r'<[^>]+>', result):
        raise ValueError('Word formatting markup changed')
    return result


def verify_semantics(original, revised):
    before, after = table_cells(original), table_cells(revised)
    book_fields = fields()
    changed, checked = 0, 0
    for ti, (old_table, new_table) in enumerate(zip(before, after, strict=True)):
        for ri, (old_row, new_row) in enumerate(zip(old_table, new_table, strict=True)):
            for ci, (old_cell, new_cell) in enumerate(zip(old_row, new_row, strict=True)):
                old, new = old_cell[0], new_cell[0]
                if ti in (15, 16) and ri and ci in CODE_COLUMNS:
                    field = book_fields[CODE_COLUMNS.index(ci)]
                    legacy = {c['legacy_code']: c['machine_value'] for c in field['categories']}
                    current = {c['new_code']: c['machine_value'] for c in field['categories']}
                    if legacy.get(old) != current.get(new):
                        raise ValueError(f'Decoded category changed at table {ti}, row {ri}, column {ci}')
                    checked += 1
                    changed += old != new
                elif ti == 1 and ri and ci == 1:
                    continue
                elif old != new:
                    raise ValueError(f'Non-code cell changed at table {ti}, row {ri}, column {ci}')
    return {'categorical_cells_verified': checked, 'categorical_cells_changed': changed,
            'non_code_table_cells_exact': True, 'formatting_markup_exact': True}


def migrate(source, destination, reference):
    source, destination, reference = map(Path, (source, destination, reference))
    if destination.resolve() in {source.resolve(), reference.resolve()}:
        raise ValueError('Write the revised document to a separate path to preserve its source')
    if hashlib.sha256(reference.read_bytes()).hexdigest() != SOURCE_HASH:
        raise ValueError('Original supplementary SHA-256 does not match the verified source')
    with ZipFile(reference) as original, ZipFile(source) as incoming:
        before = original.read(MEMBER).decode('utf-8')
        after = revised_xml(before)
        if incoming.namelist() != original.namelist():
            raise ValueError('Unknown document package structure')
        for name in original.namelist():
            if name != MEMBER and incoming.read(name) != original.read(name):
                raise ValueError(f'Unrelated document package member changed: {name}')
        current = incoming.read(MEMBER).decode('utf-8')
        if current == before:
            detected = 'legacy_codes_v1'
        elif current == after:
            detected = PUBLIC_VERSION
        else:
            raise ValueError('Mixed, ambiguous or unrelated document; no complete codebook version matches')
        report = verify_semantics(before, after)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if detected == PUBLIC_VERSION:
            content = source.read_bytes()
        else:
            content = None
            with ZipFile(destination, 'w') as revised:
                for member in incoming.infolist():
                    revised.writestr(member, after.encode('utf-8') if member.filename == MEMBER
                                     else incoming.read(member.filename))
    if content is not None:
        destination.write_bytes(content)
    with ZipFile(reference) as original, ZipFile(destination) as revised:
        if any(original.read(name) != revised.read(name) for name in original.namelist() if name != MEMBER):
            raise ValueError('Unrelated package content changed during writing')
    return {**report, 'detected_input_version': detected, 'output_version': PUBLIC_VERSION,
            'changes_applied_this_run': report['categorical_cells_changed'] if detected != PUBLIC_VERSION else 0,
            'source_sha256': SOURCE_HASH, 'output_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def numeric_equal(display, canonical):
    return canonical == '' if display == '—' else canonical != '' and Decimal(display) == Decimal(canonical)


def publication_rows(xml, target, root=ROOT):
    """Match complete Word observations to canonical identities without assuming row-number identity."""
    table = table_cells(xml)[15 if target == 'tnf' else 16]
    book_fields = fields()
    identities = {row['dataset_row_id']: row['DOI'] for row in read_rows(root / 'data/record_identity.csv')
                  if row['target'] == target}
    dataset = read_rows(root / f'data/{target}.csv')
    records, used, doi = [], set(), None
    for row in table[1:]:
        values = [cell[0] for cell in row]
        doi = values[0] or doi
        decoded = {field['machine_column']: {c['new_code']: c['machine_value'] for c in field['categories']}[values[column]]
                   for column, field in zip(CODE_COLUMNS, book_fields, strict=True)}
        matches = [record for record in dataset if identities[record['dataset_row_id']] == doi
                   and all(record[key] == value for key, value in decoded.items())
                   and all(numeric_equal(values[column], record[key]) for column, key in NUMERIC_COLUMNS.items())]
        if len(matches) != 1 or matches[0]['dataset_row_id'] in used:
            raise ValueError(f'{target} supplementary row {values[1]} has no unique unused canonical match')
        identity = matches[0]['dataset_row_id']
        used.add(identity)
        record = {VERSION_COLUMN: PUBLIC_VERSION, 'dataset_row_id': identity,
                  'DOI': values[0], 'supplementary_no': values[1]}
        for column in range(2, 13):
            key = NUMERIC_COLUMNS.get(column) or book_fields[CODE_COLUMNS.index(column)]['machine_column']
            record[key] = values[column]
        records.append(record)
    if used != {row['dataset_row_id'] for row in dataset}:
        raise ValueError('Supplementary rows do not cover the complete canonical dataset')
    return records


def export_tables(document, output, root=ROOT):
    with ZipFile(document) as archive:
        xml = archive.read(MEMBER).decode('utf-8')
    with ZipFile(root / 'docs/source/Supplementary_01Oct26_YL.docx') as reference:
        expected = revised_xml(reference.read(MEMBER).decode('utf-8'))
    if xml != expected:
        raise ValueError('Publication export requires the complete verified new document version')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checks = []
    for target in ('tnf', 'il10'):
        records = publication_rows(xml, target, root)
        path = output / f'{target}_supplementary_codes_20261019.csv'
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        canonical = {row['dataset_row_id']: row for row in read_rows(root / f'data/{target}.csv')}
        differences, missingness = [], []
        for row in records:
            for key in NUMERIC_COLUMNS.values():
                displayed, original = row[key], canonical[row['dataset_row_id']][key]
                missingness.append((displayed == '—') == (original == ''))
                if original != '':
                    differences.append(abs(Decimal(displayed) - Decimal(original)))
        checks.extend([
            {'check': target + '_unique_canonical_records_matched', 'value': len(records)},
            {'check': target + '_maximum_displayed_numeric_difference', 'value': max(differences)},
            {'check': target + '_displayed_numeric_cells_compared', 'value': len(differences)},
            {'check': target + '_total_cells_checked_for_missingness', 'value': len(missingness)},
            {'check': target + '_missingness_preserved', 'value': all(missingness)},
        ])
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'docs/source/Supplementary_01Oct26_YL.docx')
    parser.add_argument('--reference', type=Path, default=ROOT / 'docs/source/Supplementary_01Oct26_YL.docx')
    parser.add_argument('--destination', type=Path, default=ROOT / 'docs/Supplementary_frequency_ordered_20261019.docx')
    parser.add_argument('--exports', type=Path, default=ROOT / 'data/publication')
    parser.add_argument('--audit-output', type=Path, default=ROOT / 'results/publication_codes_20261019')
    args = parser.parse_args()
    report = migrate(args.source, args.destination, args.reference)
    checks = export_tables(args.destination, args.exports)
    args.audit_output.mkdir(parents=True, exist_ok=True)
    with (args.audit_output / 'document_semantic_checks.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['check', 'value'])
        writer.writeheader()
        writer.writerows([{'check': key, 'value': value} for key, value in report.items()] + checks)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
