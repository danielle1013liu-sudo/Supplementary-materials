import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import ROOT, read_csv
from src.hydrogel_immune_ml.public_encoding import fields
from src import plot_interpretation as plots


def test_importance_uses_exported_values_without_consensus_inputs(tmp_path, monkeypatch):
    exported = read_csv(ROOT / 'outputs/shap/test_importance.csv')
    source = tmp_path / 'shap'
    source.mkdir()
    exported.to_csv(source / 'test_importance.csv', index=False)
    captured = []
    monkeypatch.setattr(plots, 'save_figure', lambda fig, *_: captured.append(fig))
    plots.plot_importance(tmp_path, tmp_path / 'figures')
    figure = captured[0]
    try:
        for ax, target in zip(figure.axes, plots.TARGETS):
            expected = exported.loc[exported.target.eq(target)].sort_values('consensus_mean_abs_SHAP_pg_ml')
            np.testing.assert_array_equal(
                [bar.get_width() for bar in ax.patches], expected.consensus_mean_abs_SHAP_pg_ml,
            )
            assert f'n = {expected.n_Test.iloc[0]}' in ax.get_title()
    finally:
        plt.close(figure)


def test_landscape_legends_follow_the_shared_publication_order():
    codebook = {field['machine_column']: field['categories'] for field in fields()}
    for feature, categories in codebook.items():
        values = pd.DataFrame({feature: [c['machine_value'] for c in categories],
                               'tsne_1': 0, 'tsne_2': 0})
        figure, ax = plt.subplots()
        try:
            plots.color_landscape(ax, values, feature, codebook)
            labels = [label.get_text() for label in ax.get_legend().get_texts()]
            assert labels == [f"{c['new_code']}: {plots.CATEGORY_LABELS[c['machine_value']]}"
                              for c in categories]
        finally:
            plt.close(figure)
