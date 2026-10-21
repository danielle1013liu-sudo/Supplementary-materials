import numpy as np
import pytest
from src.data import feature_names, load_cases, load_data
from src.hydrogel_immune_ml.inference import form_frame, load_resources, predict_form, predict_frame
from src.pareto.common import csv, DATA
from src.hydrogel_immune_ml.application_schema import fitted_columns


@pytest.fixture(scope='module')
def resources():
    return load_resources()


def case_form():
    row = load_cases('tnf')[feature_names()].iloc[0]
    return {'values': {key: str(value) for key, value in row.items()}, 'unreported': {}}


def test_actual_form_precision_and_downloads(resources, tmp_path):
    form = case_form()
    form['values']['modulus_value_kPa'] = '1.2345678901234567'
    form['unreported']['pore_size_um'] = True
    frame = form_frame(form)
    assert frame.modulus_value_kPa.iloc[0] == float('1.2345678901234567')
    assert np.isnan(frame.pore_size_um.iloc[0])
    result = predict_form(form, resources, tmp_path)
    saved = csv(tmp_path / 'predictions.csv')
    for target, resource in resources.items():
        actual = saved.loc[saved.target.eq(target) & ~saved.realisation.eq('mean'), 'prediction_pg_ml'].to_numpy()
        expected = np.array([pipeline.predict(frame)[0] for _, pipeline in resource['models']])
        np.testing.assert_array_equal(actual, expected)
    assert result['downloads']['inputs'] == (tmp_path / 'inputs.csv').read_text()
    assert (tmp_path / 'contributions.csv').is_file()


@pytest.mark.parametrize('row', [0, 1])
def test_primary_case_predictions_and_shap(resources, row):
    frame = load_cases('tnf')[feature_names()].iloc[row:row+1]
    predictions, contributions, warnings = predict_frame(frame, resources)
    for target, resource in resources.items():
        actual = predictions.loc[predictions.target.eq(target) & ~predictions.realisation.eq('mean'), 'prediction_pg_ml'].to_numpy()
        expected = np.array([pipeline.predict(frame)[0] for _, pipeline in resource['models']])
        np.testing.assert_array_equal(actual, expected)
    assert contributions.additivity_residual_pg_ml.abs().max() < 1e-7
    assert not warnings


def test_applicability_matches_released_representative(resources):
    reference = csv(DATA / 'current/results/FINAL_PARETO_REPRESENTATIVES.csv').iloc[:1]
    predictions, _, _ = predict_frame(reference[feature_names()], resources)
    for target in ('tnf', 'il10'):
        actual = predictions.loc[predictions.target.eq(target), 'AD_ratio'].iloc[0]
        np.testing.assert_allclose(actual, reference[target + '_AD_ratio'].iloc[0], rtol=0, atol=1e-10)


def test_unseen_category_reported_without_other_substitution(resources):
    frame = load_data('tnf')[feature_names()].iloc[:1].copy()
    frame['macrophage_model_class'] = 'BMDM'
    _, _, warnings = predict_frame(frame, resources)
    assert sum('IL10 seed' in warning and 'BMDM' in warning for warning in warnings) == 3
    assert any('IL10 applicability encoder' in warning and 'BMDM' in warning for warning in warnings)
    frame['macrophage_model_class'] = 'new_unrecognised_cell'
    with pytest.raises(ValueError, match='Unknown category'):
        predict_frame(frame, resources)


@pytest.mark.parametrize('value', ['', 'nan', 'inf', '1x', '-2'])
def test_malformed_numeric_form_rejected(value):
    form = case_form()
    form['values']['modulus_value_kPa'] = value
    with pytest.raises(ValueError):
        form_frame(form)


def test_blank_category_cannot_be_imputed():
    form = case_form()
    form['values']['polymer_origin_class'] = ''
    with pytest.raises(ValueError, match='blank choice'):
        form_frame(form)


def test_current_models_use_median_and_keep_fitted_coordinates(resources):
    assert [s for s, _ in resources['il10']['models']] == [18571, 28094, 30787]
    for target, resource in resources.items():
        for seed, pipeline in resource['models']:
            preprocessor = pipeline.named_steps['preprocess']
            assert preprocessor.named_transformers_['numerical'].named_steps['imputer'].strategy == 'median'
            identities = fitted_columns(preprocessor)
            assert identities == resource['model_columns'][seed]
            assert len(identities) == preprocessor.transform(load_cases(target)[feature_names()]).shape[1]


def test_missing_numeric_predictions_and_summary_match_saved_pipeline(resources, tmp_path):
    form = case_form()
    names = ['modulus_value_kPa', 'pore_size_um', 'culture_or_exposure_duration_h']
    form['unreported'] = dict.fromkeys(names, True)
    frame = form_frame(form)
    result = predict_form(form, resources, tmp_path)
    for row in result['targets']:
        expected = np.array([pipe.predict(frame)[0] for _, pipe in resources[row['target']]['models']])
        assert row['mean'] == float(expected.mean())
        assert row['sd'] == float(expected.std(ddof=1))


def test_loading_and_prediction_do_not_fit(monkeypatch):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    def forbid(*args, **kwargs):
        raise AssertionError('Application inference must not fit an estimator')

    for estimator in (Pipeline, ColumnTransformer):
        monkeypatch.setattr(estimator, 'fit', forbid)
        monkeypatch.setattr(estimator, 'fit_transform', forbid)
    resources = load_resources()
    predict_frame(form_frame(case_form()), resources)
