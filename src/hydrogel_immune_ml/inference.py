"""Application inference in the frozen model and applicability coordinates."""

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits

from ..data import ROOT, check_environment, check_inputs, feature_names, file_hash, read_json, split_data, value_hash
from ..models import load_models
from .application_schema import fitted_columns, unsupported_categories, validate_application_input
from .public_encoding import decode_public_input


def load_resources(runtime='reproduction'):
    if runtime == 'reproduction':
        check_environment()
    elif runtime == 'community-cloud':
        from .application_runtime import check_cloud_environment
        check_cloud_environment()
    else:
        raise ValueError(f'Unknown application runtime: {runtime}')
    check_inputs()
    release = read_json(ROOT / 'data/pareto/CURRENT_RELEASE.json')
    for section in ('source_sha256', 'snapshot_sha256'):
        for name, expected in release[section].items():
            if file_hash(ROOT / name) != expected:
                raise ValueError(f'Current release checksum differs: {name}')
    resources = {}
    settings_path = ROOT / release['snapshot_directory'] / 'inputs/APPLICABILITY_DOMAIN_SETTINGS.json'
    for settings in read_json(settings_path):
        target = settings['target']
        path = ROOT / settings['path'].replace('\\', '/')
        if file_hash(path) != settings['sha256']:
            raise ValueError(f'Diagnostic preprocessor checksum differs: {target}')
        training = split_data(target)[0]
        if training['dataset_row_id'].tolist() != settings['Training_IDs']:
            raise ValueError(f'Diagnostic Training membership differs: {target}')
        if value_hash(settings['Training_IDs']) != settings['Training_ID_hash']:
            raise ValueError(f'Diagnostic Training identity checksum differs: {target}')
        if file_hash(ROOT / f'data/{target}.csv') != settings['dataset_sha256']:
            raise ValueError(f'Diagnostic Training dataset checksum differs: {target}')
        preprocessor = joblib.load(path)
        transformed = preprocessor.transform(training[feature_names()])
        if value_hash(transformed.tolist()) != settings['transformed_Training_hash']:
            raise ValueError(f'Diagnostic Training coordinates differ: {target}')
        pipelines = load_models(target)
        if [seed for seed, _ in pipelines] != release['accepted_seeds'][target]:
            raise ValueError(f'Current release seed identity differs: {target}')
        resources[target] = {
            'models': pipelines,
            'model_columns': {seed: fitted_columns(pipe.named_steps['preprocess']) for seed, pipe in pipelines},
            'domain_preprocessor': preprocessor,
            'domain_training': transformed,
            'domain_threshold': settings['threshold_q95'],
            'domain_neighbors': settings['neighbors'],
            'domain_columns': fitted_columns(preprocessor),
        }
    if runtime == 'community-cloud':
        from .application_runtime import verify_cloud_predictions
        verify_cloud_predictions(resources)
    return resources


def form_frame(form):
    values = dict(form['values'])
    numeric = read_json(ROOT / 'data/feature_schema.json')['numeric_semantics']
    for field, details in numeric.items():
        if form.get('unreported', {}).get(field, False):
            values[field] = np.nan
        elif values.get(field, '') == '':
            raise ValueError(f'Enter {details["display_name"]} or select Unreported')
        else:
            try:
                values[field] = float(values[field])
            except (TypeError, ValueError) as error:
                raise ValueError(f'{details["display_name"]} must be a number') from error
            if not np.isfinite(values[field]):
                raise ValueError(f'{details["display_name"]} must be finite')
    frame = validate_application_input(pd.DataFrame([values]))
    return decode_public_input(frame, 'canonical')


def predict_frame(frame, resources):
    frame = validate_application_input(frame)
    if len(frame) != 1:
        raise ValueError('The application expects one submitted profile')
    predictions, contributions, warnings = [], [], []
    with threadpool_limits(limits=1):
        for target, resource in resources.items():
            support = {}
            for field, category in resource['domain_columns']:
                if category is not None:
                    support.setdefault(field, set()).add(category)
            for field, categories in support.items():
                if frame[field].iloc[0] not in categories:
                    warnings.append(f'{target.upper()} applicability encoder: {field} category {frame[field].iloc[0]} was unseen.')
            z_domain = resource['domain_preprocessor'].transform(frame)
            distances = pairwise_distances(z_domain, resource['domain_training'])
            ratio = float(np.sort(distances, axis=1)[:, :resource['domain_neighbors']].mean() / resource['domain_threshold'])
            target_values, grouped_values, bases = [], [], []
            for seed, pipeline in resource['models']:
                identities = resource['model_columns'][seed]
                unknown = unsupported_categories(frame, identities)
                for field, values in unknown.items():
                    warnings.append(f'{target.upper()} seed {seed}: {field} category {values[0]} was unseen by this fitted encoder.')
                value = float(pipeline.predict(frame)[0])
                target_values.append(value)
                predictions.append({'target': target, 'realisation': str(seed), 'prediction_pg_ml': value, 'sample_SD_pg_ml': np.nan, 'AD_ratio': ratio, 'physically_invalid': value < 0})
                z = pipeline.named_steps['preprocess'].transform(frame)
                explainer = shap.TreeExplainer(pipeline.named_steps['model'], data=None, feature_perturbation='tree_path_dependent', model_output='raw')
                phi = np.asarray(explainer.shap_values(z, approximate=False, check_additivity=True))[0]
                base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
                grouped = {field: 0.0 for field in feature_names()}
                for (field, _), contribution in zip(identities, phi, strict=True):
                    grouped[field] += float(contribution)
                np.testing.assert_allclose(base + sum(grouped.values()), value, atol=1e-7, rtol=0)
                grouped_values.append(list(grouped.values()))
                bases.append(base)
                for field, contribution in grouped.items():
                    contributions.append({'target': target, 'realisation': str(seed), 'descriptor': field, 'SHAP_pg_ml': contribution, 'base_pg_ml': base, 'prediction_pg_ml': value, 'additivity_residual_pg_ml': base + sum(grouped.values()) - value})
            mean, sd = float(np.mean(target_values)), float(np.std(target_values, ddof=1))
            predictions.append({'target': target, 'realisation': 'mean', 'prediction_pg_ml': mean, 'sample_SD_pg_ml': sd, 'AD_ratio': ratio, 'physically_invalid': any(v < 0 for v in target_values)})
            mean_phi = np.mean(grouped_values, axis=0)
            mean_base = float(np.mean(bases))
            np.testing.assert_allclose(mean_base + mean_phi.sum(), mean, atol=1e-7, rtol=0)
            for field, contribution in zip(feature_names(), mean_phi, strict=True):
                contributions.append({'target': target, 'realisation': 'mean', 'descriptor': field, 'SHAP_pg_ml': float(contribution), 'base_pg_ml': mean_base, 'prediction_pg_ml': mean, 'additivity_residual_pg_ml': float(mean_base + mean_phi.sum() - mean)})
    return pd.DataFrame(predictions), pd.DataFrame(contributions), warnings


def predict_form(form, resources, output):
    frame = form_frame(form)
    predictions, contributions, warnings = predict_frame(frame, resources)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in [('inputs', frame), ('predictions', predictions), ('contributions', contributions)]:
        table.to_csv(output / f'{name}.csv', index=False, float_format='%.17g')
    # The interface reads saved numerical outputs; displayed plots never recompute them.
    predictions = pd.read_csv(output / 'predictions.csv', float_precision='round_trip')
    contributions = pd.read_csv(output / 'contributions.csv', float_precision='round_trip')
    targets = []
    for target in ('tnf', 'il10'):
        rows = predictions.loc[predictions.target.eq(target)]
        mean = rows.loc[rows.realisation.eq('mean')].iloc[0]
        targets.append({'target': target, 'mean': float(mean.prediction_pg_ml), 'sd': float(mean.sample_SD_pg_ml), 'domain_ratio': float(mean.AD_ratio), 'invalid': bool(mean.physically_invalid), 'individual': [{'seed': row.realisation, 'value': float(row.prediction_pg_ml)} for row in rows.itertuples() if row.realisation != 'mean'], 'contributions': contributions.loc[contributions.target.eq(target) & contributions.realisation.eq('mean'), ['descriptor', 'SHAP_pg_ml']].to_dict('records')})
    return {'targets': targets, 'warnings': warnings, 'downloads': {name: (output / f'{name}.csv').read_text(encoding='utf-8') for name in ('inputs', 'predictions', 'contributions')}}
