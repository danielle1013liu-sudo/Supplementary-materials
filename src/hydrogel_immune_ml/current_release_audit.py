"""Replay the current saved release and write website verification tables."""

from pathlib import Path
import numpy as np
import pandas as pd

from ..data import ROOT, feature_names, load_cases, load_data, read_json, split_data
from .inference import load_resources, predict_frame


def save(frame, path):
    frame.to_csv(path, index=False, float_format='%.17g')


def prepare_display_ranges(output):
    pooled = pd.concat([load_data(target) for target in ('tnf', 'il10')])
    numeric = read_json(ROOT / 'data/feature_schema.json')['numeric_semantics']
    save(pd.DataFrame([{'field': name, 'minimum': pooled[name].min(),
                       'maximum': pooled[name].max(),
                       'reported_endpoint_records': pooled[name].notna().sum()}
                      for name in numeric]), output / 'numeric_display_ranges.csv')


def main(output=None):
    output = Path(output) if output else ROOT / 'results/revision_current'
    output.mkdir(parents=True, exist_ok=True)
    prepare_display_ranges(output)
    resources = load_resources()
    frozen = np.load(ROOT / 'data/reference/accepted_predictions.npz')
    checks, identities = [], []
    for target, resource in resources.items():
        training, test = split_data(target)
        scopes = [('Training', training), ('Test', test), ('C4', load_cases(target))]
        for seed, pipeline in resource['models']:
            model = pipeline.named_steps['model']
            identities.append({'target': target, 'seed': seed,
                               'estimator': type(model).__name__,
                               'numeric_imputer': pipeline.named_steps['preprocess'].named_transformers_['numerical'].named_steps['imputer'].strategy})
            for scope, frame in scopes:
                actual = pipeline.predict(frame[feature_names()])
                expected = frozen[f'{target}_{seed}_{scope}']
                ids = frame['dataset_row_id'] if 'dataset_row_id' in frame else frame['hydrogel_group']
                save(pd.DataFrame({'row_identity': ids,
                                   'actual_pg_ml': actual, 'release_pg_ml': expected,
                                   'difference_pg_ml': actual - expected}),
                     output / f'{target}_{seed}_{scope}_replay.csv')
                np.testing.assert_array_equal(actual, expected)
                checks.append({'check': f'{target}_{seed}_{scope}_exact', 'passed': True})
    save(pd.DataFrame(identities), output / 'model_identities.csv')
    cases = load_cases('tnf')[feature_names()]
    release = read_json(ROOT / 'data/pareto/CURRENT_RELEASE.json')
    representatives = pd.read_csv(ROOT / release['reference_directory'] / 'FINAL_PARETO_REPRESENTATIVES.csv', float_precision='round_trip')
    profiles = [(f'case_{i}', cases.iloc[i:i + 1]) for i in range(len(cases))]
    profiles += [(f'representative_{i}', representatives.iloc[i:i + 1][feature_names()]) for i in range(len(representatives))]
    missing = cases.iloc[:1].copy()
    missing[['modulus_value_kPa', 'pore_size_um', 'culture_or_exposure_duration_h']] = np.nan
    profiles.append(('numeric_unreported', missing))
    all_predictions, all_contributions, all_warnings = [], [], []
    for profile, frame in profiles:
        predictions, contributions, warnings = predict_frame(frame, resources)
        predictions.insert(0, 'profile', profile)
        contributions.insert(0, 'profile', profile)
        all_predictions.append(predictions)
        all_contributions.append(contributions)
        all_warnings.extend({'profile': profile, 'warning': warning} for warning in warnings)
        for target, resource in resources.items():
            expected = np.array([pipe.predict(frame)[0] for _, pipe in resource['models']])
            rows = predictions.loc[predictions.target.eq(target)]
            np.testing.assert_array_equal(rows.loc[rows.realisation.ne('mean'), 'prediction_pg_ml'], expected)
            summary = rows.loc[rows.realisation.eq('mean')].iloc[0]
            assert summary.prediction_pg_ml == float(expected.mean())
            assert summary.sample_SD_pg_ml == float(expected.std(ddof=1))
            if profile.startswith('representative_'):
                index = int(profile.rsplit('_', 1)[1])
                np.testing.assert_allclose(summary.AD_ratio, representatives.iloc[index][target + '_AD_ratio'], rtol=0, atol=1e-10)
            checks.append({'check': f'{profile}_{target}_prediction_summary', 'passed': True})
    save(pd.concat(all_predictions, ignore_index=True), output / 'application_predictions.csv')
    save(pd.concat(all_contributions, ignore_index=True), output / 'application_contributions.csv')
    save(pd.DataFrame(all_warnings, columns=['profile', 'warning']), output / 'application_warnings.csv')
    persisted = pd.read_csv(output / 'application_contributions.csv', float_precision='round_trip')
    assert persisted.additivity_residual_pg_ml.abs().max() < 1e-7
    checks.append({'check': 'persisted_grouped_SHAP_additivity', 'passed': True})
    save(pd.DataFrame(checks), output / 'backend_checks.csv')
    print(f'Current saved-release replay passed: {len(checks)} checks; no fitting performed.')


if __name__ == '__main__':
    main()
