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
        verify_release_document(document, root)
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


def legacy_main():
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





RELEASE_VERSION = 'crosslink7_rfmedian_20261020'
CURRENT_SOURCE_HASH = '415e4731c52fa2ec59583d0dc96ef16c65fc50fdd6b45fc6e6322d2567c16275'
ROW_MAP_HASH = '6febe77f8331f7012ba3b9d2cde99ee42992060a55f2fa7cc61cb14aeb564315'
IL10_HASH = '87ec0a562440edb7a2138fba7f01385848241e2cec5c6872b69031be3bb54188'
CURRENT_SOURCE = 'docs/source/Supplementary_frequency_ordered_20261019.docx'
CURRENT_DOCUMENT = 'docs/Supplementary_IL10_updated_20261020.docx'
RELEASE_MANIFEST = 'docs/supplementary_release_20261020.json'
ROW_MAP = 'docs/source/publication_row_identities_20261019.json'
CORRECTED_IDS = {'IL10-0003', 'IL10-0004', 'IL10-0005', 'IL10-0016',
                 'IL10-0017', 'IL10-0018', 'IL10-0078'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _release_input_path(root, name, digest):
    """Resolve missing replay tables to the identical published Pareto snapshot."""
    path = root / name
    replay_tables = {
        'outputs/pareto/results/FINAL_PARETO_PROFILES.csv',
        'outputs/pareto/results/FINAL_PARETO_REPRESENTATIVES.csv',
    }
    if path.exists() or name not in replay_tables:
        return path
    snapshot = json.loads((root / 'data/pareto/CURRENT_RELEASE.json').read_text(encoding='utf-8'))
    directory = 'data/pareto/current/results'
    published = directory + '/' + Path(name).name
    if (snapshot['reference_directory'] != directory
            or snapshot['snapshot_sha256'].get(published) != digest):
        raise ValueError(f'Published Pareto table does not match the document manifest: {name}')
    for source, expected in snapshot['source_sha256'].items():
        if sha256(root / source) != expected:
            raise ValueError(f'Accepted source differs from the Pareto snapshot: {source}')
    return root / published


def verify_release_document(document, root=ROOT):
    """Accept the current document only with its exact source/result manifest."""
    record = json.loads((root / RELEASE_MANIFEST).read_text(encoding='utf-8'))
    if record['release_version'] != RELEASE_VERSION or sha256(document) != record['document_sha256']:
        raise ValueError('Supplementary document does not match the frozen release manifest')
    for name, digest in record['inputs_sha256'].items():
        if sha256(_release_input_path(root, name, digest)) != digest:
            raise ValueError(f'Supplementary input changed after generation: {name}')
    return record


def _set_text(container, text):
    """Replace text while retaining the first run's local typography."""
    paragraph = container.paragraphs[0] if hasattr(container, 'paragraphs') else container
    if hasattr(container, 'paragraphs'):
        for extra in list(container.paragraphs[1:]):
            extra._p.getparent().remove(extra._p)
    runs = paragraph.runs
    if runs:
        runs[0].text = str(text)
        for run in runs[1:]:
            run._r.getparent().remove(run._r)
    else:
        paragraph.add_run(str(text))
    caption = re.match(r'((?:Table|Figure) S[0-9]+[a-z]?(?: \(continued\))?\.)\s+', str(text))
    if caption and not hasattr(container, 'paragraphs'):
        from copy import deepcopy
        first = paragraph.runs[0]
        first.text = caption.group(1) + ' '
        first.bold = True
        rest = paragraph.add_run(str(text)[caption.end():])
        if first._r.rPr is not None:
            rest._r.insert(0, deepcopy(first._r.rPr))
        rest.bold = False


def _fill_rows(table, rows):
    from copy import deepcopy
    from docx.oxml import OxmlElement
    template = deepcopy(table.rows[1]._tr)
    for row in list(table.rows)[1:]:
        table._tbl.remove(row._tr)
    for values in rows:
        table._tbl.append(deepcopy(template))
        row = table.rows[-1]
        if len(row.cells) != len(values):
            raise ValueError('Column mismatch when updating supplementary table')
        for cell, value in zip(row.cells, values, strict=True):
            _set_text(cell, value)
    header = table.rows[0]._tr.get_or_add_trPr()
    if not header.findall('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tblHeader'):
        header.append(OxmlElement('w:tblHeader'))
    for row in table.rows:
        for height in row._tr.xpath('./w:trPr/w:trHeight'):
            height.getparent().remove(height)


def _domain_table(table, profiles, start):
    """Use a vertical four-column layout so any Pareto-front size stays readable."""
    from copy import deepcopy
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches
    templates = [deepcopy(table.rows[i]._tr) for i in (0, 1)]
    for row in templates:
        for cell in list(row.findall(qn('w:tc')))[4:]: row.remove(cell)
    for row in list(table.rows): table._tbl.remove(row._tr)
    table._tbl.append(templates[0]); table._tbl.append(templates[1])
    grid = table._tbl.tblGrid
    for child in list(grid): grid.remove(child)
    for width in [1.0, 2.8, 2.8, 3.5]:
        col = OxmlElement('w:gridCol'); col.set(qn('w:w'), str(round(width * 1440))); grid.append(col)
    for row in table.rows:
        for cell, width in zip(row.cells, [1.0, 2.8, 2.8, 3.5], strict=True): cell.width = Inches(width)
    for cell, label in zip(table.rows[0].cells, ['Profile No.', 'TNF-α domain ratio', 'IL-10 domain ratio', 'Non-dominance frequency'], strict=True):
        _set_text(cell, label)
    _fill_rows(table, [[start + i, f"{p['tnf_AD_ratio']:.3f}", f"{p['il10_AD_ratio']:.3f}",
                        f"{int(p['nondomination_frequency_of_9'])}/9"] for i, p in enumerate(profiles)])


def _replace_picture(document, paragraph, path):
    from docx.shared import Inches
    from PIL import Image
    for run in list(paragraph.runs):
        run._r.getparent().remove(run._r)
    with Image.open(path) as image:
        ratio = image.height / image.width
    # Preserve the landscape page's usable width, with room for a caption.
    width = min(10.15, 5.65 / ratio)
    picture = paragraph.add_run().add_picture(str(path), width=Inches(width))
    picture._inline.docPr.set('descr', path.stem.replace('_', ' '))


def build_release(root=ROOT, destination=None):
    """Regenerate affected tables and figures from the accepted release artifacts.

    No fitting or inference occurs here. Source identity, the seven corrected cells,
    model configuration, source tables and figure files are checked and recorded.
    """
    import math
    import pandas as pd
    from docx import Document
    from .publication_exports import refresh_counts, verify_counts, export_canonical_codes, export_manifest
    root = Path(root)
    if root.resolve() != ROOT.resolve():
        raise ValueError('Run the builder from its repository checkout')
    destination = Path(destination or root / CURRENT_DOCUMENT)
    if destination.resolve() == (root / CURRENT_SOURCE).resolve():
        raise ValueError('The historical source must remain unchanged')
    for name, expected in [(CURRENT_SOURCE, CURRENT_SOURCE_HASH), (ROW_MAP, ROW_MAP_HASH),
                            ('data/il10.csv', IL10_HASH)]:
        if sha256(root / name) != expected:
            raise ValueError(f'Unrecognised release input: {name}')
    parameters = json.loads((root / 'data/model_parameters.json').read_text())
    il10 = parameters['targets']['il10']
    if il10['seeds'] != [18571, 28094, 30787] or il10['numeric_imputer'].get('strategy') != 'median':
        raise ValueError('Expected the accepted RF/median IL-10 release')
    inputs = [CURRENT_SOURCE, ROW_MAP, 'data/tnf.csv', 'data/il10.csv', 'data/record_identity.csv',
              'data/model_parameters.json', 'data/software_versions.json',
              'data/validation_evidence/final_seed_all20_comparisons.csv',
              'outputs/case_studies/summary.csv', 'outputs/tsne/diagnostics.csv',
              'outputs/pareto/results/FINAL_PARETO_PROFILES.csv',
              'outputs/pareto/results/FINAL_PARETO_REPRESENTATIVES.csv']
    figures = ['parity_predictions', 'tnf_shap_landscape', 'il10_shap_landscape', 'context_shap_landscapes']
    inputs += [f'outputs/figures/{f}.png' for f in figures]
    # Missing result files are a hard error; stale embedded figures are never retained.
    before_hashes = {name: sha256(root / name) for name in inputs}
    document = Document(root / CURRENT_SOURCE)
    tables, paragraphs = document.tables, document.paragraphs
    row_map = json.loads((root / ROW_MAP).read_text())
    changed = []
    for target, index in [('tnf', 15), ('il10', 16)]:
        canonical = {r['dataset_row_id']: r for r in read_rows(root / f'data/{target}.csv')}
        for mapping in row_map[target]:
            row = tables[index].rows[int(mapping['supplementary_no'])]
            record = canonical[mapping['dataset_row_id']]
            for column, field in zip(CODE_COLUMNS, fields(), strict=True):
                code = {c['machine_value']: c['new_code'] for c in field['categories']}[record[field['machine_column']]]
                if row.cells[column].text != code:
                    if target != 'il10' or column != 7 or record['dataset_row_id'] not in CORRECTED_IDS:
                        raise ValueError('Unexpected categorical change in source dataset table')
                    changed.append(record['dataset_row_id'])
                    _set_text(row.cells[column], code)
    if set(changed) != CORRECTED_IDS or len(changed) != 7:
        raise ValueError('Expected exactly seven crosslink-level corrections')
    comparison = pd.read_csv(root / inputs[7])
    family_names = {'LightGBM': 'LGBMRegressor', 'Gradient boosting regression': 'GradientBoostingRegressor',
                    'XGBoost': 'XGBRegressor', 'Random forest': 'RandomForestRegressor'}
    imputers = {'Median': 'median', 'Mean': 'mean', 'Iterative Bayesian ridge': 'iterative_bayesian_ridge',
                'Scaled KNN (k = 3)': 'scaled_knn_3', 'Scaled KNN (k = 5)': 'scaled_knn_5'}
    # The frozen Training-fold assignments are shared by every comparison.
    for target, table_index in [('tnf', 5), ('il10', 6)]:
        for row in tables[table_index].rows[1:]:
            subset = comparison.loc[(comparison.target == target) &
                (comparison.family == family_names[row.cells[3].text]) &
                (comparison.imputer == imputers[row.cells[2].text])]
            if len(subset) != 3 or set(subset.estimator_seed) != set(parameters['targets'][target]['seeds']):
                raise ValueError('Comparison is not the same three accepted realisations')
            for column, metric, digits in [(4, 'Validation_R2', 3), (5, 'Validation_RMSE', 2), (6, 'Validation_MAE', 2)]:
                _set_text(row.cells[column], f'{subset[metric].mean():.{digits}f}')
        chosen = comparison.loc[(comparison.target == target) &
                 (comparison.family == parameters['targets'][target]['family']) & (comparison.imputer == 'median')]
        if len(chosen) != 3: raise ValueError('Selected model comparison is incomplete')
        row = tables[7].rows[1 if target == 'tnf' else 2]
        for column, metric, digits in [(3, 'Validation_R2', 3), (4, 'Validation_RMSE', 2), (5, 'Validation_MAE', 2)]:
            _set_text(row.cells[column], f'{chosen[metric].mean():.{digits}f} ± {chosen[metric].std(ddof=1):.{digits}f}')
    _set_text(tables[8].rows[2].cells[2], 'Median')
    config_rows = {7: 'n_estimators', 9: 'max_depth', 11: 'min_samples_leaf', 12: 'min_samples_split',
                   14: 'bootstrap', 15: 'max_features', 16: 'min_impurity_decrease', 17: 'ccp_alpha', 29: 'n_jobs'}
    for row, key in config_rows.items():
        value = il10['parameters'][key]
        _set_text(tables[8].rows[row].cells[2], str(value) if not isinstance(value, float) else f'{value:g}')
    versions = json.loads((root / 'data/software_versions.json').read_text())
    settings = [
        (23, 'Numerical imputer strategy', 'Median', 'Median'),
        (24, 'Imputer statistics', 'Training rows only', 'Training rows only'),
        (25, 'Validation selection', 'Best model–imputer pair in each realisation', 'Best model–imputer pair by three-realisation mean'),
        (26, 'Implementation', f"scikit-learn {versions['scikit-learn']}", f"scikit-learn {versions['scikit-learn']}"),
        (27, 'Model version', 'Accepted Training-80 TNF release', 'Crosslink-7 RF median, 20 October 2026')]
    for index, *values in settings:
        for cell, value in zip(tables[8].rows[index].cells, values, strict=True): _set_text(cell, value)
    cases = pd.read_csv(root / 'outputs/case_studies/summary.csv')
    for row in tables[14].rows[1:]:
        for target, column in [('tnf', 2), ('il10', 4)]:
            subset = cases.loc[(cases.target == target) & (cases.formulation == row.cells[0].text)]
            if len(subset) != 1: raise ValueError('Missing or ambiguous case prediction')
            r = subset.iloc[0]
            _set_text(row.cells[column], f'{r.prediction_mean_pg_ml:.3f} ± {r.prediction_sample_SD_pg_ml:.3f}')
    profiles = pd.read_csv(root / 'outputs/pareto/results/FINAL_PARETO_PROFILES.csv').to_dict('records')
    representatives = pd.read_csv(root / 'outputs/pareto/results/FINAL_PARETO_REPRESENTATIVES.csv')
    labels = dict(zip(representatives.candidate_id, representatives.display_label))
    names = {f['machine_column']: {c['machine_value']: (c['display_label'] if f['machine_column'] in ('macrophage_model_class', 'stimulation_context') else c['display_label'].lower()) for c in f['categories']} for f in fields()}
    def display(p, key): return names.get(key, {}).get(p[key], f'{p[key]:g}' if isinstance(p[key], (float, int)) else p[key])
    material = ['polymer_origin_class', 'modulus_value_kPa', 'pore_size_um', 'material_bioactive_cue_class',
                'gelation_or_crosslinking_mechanism', 'network_density_proxy']
    context = ['macrophage_model_class', 'stimulation_context', 'cell_material_interaction_mode', 'culture_or_exposure_duration_h']
    _fill_rows(tables[9], [[i, labels.get(p['candidate_id'], '—')] + [display(p, k) for k in material] +
               [f"{p[t + '_mean_pg_ml']:.2f} ± {p[t + '_sample_SD_pg_ml']:.2f}" for t in ['tnf', 'il10']]
               for i, p in enumerate(profiles, 1)])
    _fill_rows(tables[10], [[i] + [display(p, k) for k in context] for i, p in enumerate(profiles, 1)])
    midpoint = math.ceil(len(profiles) / 2)
    _domain_table(tables[11], profiles[:midpoint], 1)
    _domain_table(tables[12], profiles[midpoint:], midpoint + 1)
    _set_text(paragraphs[7], 'Supplementary materials | revised 20 October 2026')
    _set_text(paragraphs[10], 'Table S1b. Categorical codes used in the dataset tables. Letters are nominal labels fixed in the 19 October 2026 codebook; they are not numerical or ordinal inputs to the models.')
    _set_text(paragraphs[13], 'Scores are summed across the four dimensions: 0–2, low; 3–5, medium; 6–8, high. In the revised IL-10 data, 61 records are medium (A), 64 high (B), and 8 low (C).')
    _set_text(paragraphs[22], 'Table S4c. Pooled ten-fold out-of-fold validation of the selected models. Values are mean ± sample SD across three realisations, not across ten folds.')
    _set_text(paragraphs[22], paragraphs[22].text + '\nFor IL-10, imputation and model family were selected by mean Validation R² across the same three realisations. Among the evaluated panels satisfying the Validation, Test and direction criteria, the selected panel minimised mean individual C4 MAE. The evaluated 442-realisation pool does not establish a full-range optimum. TNF-α retains the earlier per-realisation Validation-winner rule. Test and C4 data contributed to panel screening and are not independent estimates of performance after selection.')
    _set_text(paragraphs[23], '')
    _set_text(paragraphs[26], f'Table S6a. Material descriptors and predicted cytokine concentrations for the {len(profiles)} Pareto profiles shown in Fig. 10. P1–P3 denote the representative designs. Values are mean ± sample SD across three models.')
    _set_text(paragraphs[32], f'Table S6b (continued). Profiles {midpoint + 1}–{len(profiles)}.')
    _set_text(paragraphs[37], 'Predictions are mean ± sample SD across three realisations. PEG microgel predictions use separate whole-study-excluded models; HPA formulations use the primary Training-only models. C4 selection error is the mean of each model’s two absolute concentration errors; it is distinct from the absolute error in the paired concentration change.')
    _set_text(paragraphs[42], 'Figure S1. Measured versus predicted TNF-α (left) and IL-10 (right) concentrations. The upper row shows pooled ten-fold out-of-fold predictions; the lower row shows Test predictions from Training-only models. Each point is the mean of three predictions. The displayed R² is calculated from these mean predictions (ensemble R²), whereas Table S4c reports the mean and sample SD of the three individual R² values.')
    diagnostics = pd.read_csv(root / 'outputs/tsne/diagnostics.csv')
    for target, paragraph_index, figure in [('tnf', 44, 'S2'), ('il10', 46, 'S3')]:
        row = diagnostics.loc[(diagnostics.target == target) & (diagnostics.scope == 'ALL_ROWS_DESCRIPTIVE') & (diagnostics.perplexity == 30)]
        if len(row) != 1: raise ValueError('Missing t-SNE diagnostic for supplementary figure')
        caption = paragraphs[paragraph_index].text
        caption = re.sub(r'\([a-f]\) ', '', caption)
        caption = caption.replace('SHAP landscapes coloured by ', 'SHAP landscapes coloured, from left to right and top to bottom, by ')
        caption = re.sub(r'Ten-neighbour trustworthiness is [0-9.]+\.', f'Ten-neighbour trustworthiness is {row.iloc[0].trustworthiness_k10:.3f}.', caption)
        caption += ' The embedding uses all endpoint records and is descriptive; it is not an independent validation analysis.'
        _set_text(paragraphs[paragraph_index], caption)
    _set_text(paragraphs[48], 'Figure S4. SHAP landscapes coloured by macrophage model, exposure mode and exposure time (h), from left to right. The upper row shows TNF-α and the lower row IL-10. Coordinates correspond to Figures S2–S3; missing exposure-time values are shown in grey.')
    for paragraph_index, figure in zip([41, 43, 45, 47], figures, strict=True):
        _replace_picture(document, paragraphs[paragraph_index], root / f'outputs/figures/{figure}.png')
    refresh_counts()
    verify_counts(root / 'results/publication_codes_20261020')
    export_canonical_codes(root / 'data/publication')
    export_manifest(root / 'data/publication')
    inputs += ['data/category_codebook.csv', f'data/public_codebooks/{PUBLIC_VERSION}.json',
               'src/hydrogel_immune_ml/supplementary.py']
    for name, digest in before_hashes.items():
        if sha256(root / name) != digest: raise ValueError(f'Release input changed while building: {name}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    document.save(destination)
    manifest = {'release_version': RELEASE_VERSION, 'publication_codebook_version': PUBLIC_VERSION,
                'source_sha256': CURRENT_SOURCE_HASH, 'document_sha256': sha256(destination),
                'document': destination.relative_to(root).as_posix(),
                'corrected_IL10_ids': sorted(changed), 'pareto_profiles': len(profiles),
                'inputs_sha256': {name: sha256(root / name) for name in inputs}}
    (root / RELEASE_MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    checks = export_tables(destination, root / 'data/publication', root)
    verify_release_document(destination, root)
    return {**manifest, 'dataset_checks': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT / CURRENT_DOCUMENT)
    parser.add_argument('--verify', action='store_true', help='Verify existing document and all source hashes')
    args = parser.parse_args()
    result = verify_release_document(args.destination) if args.verify else build_release(destination=args.destination)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
