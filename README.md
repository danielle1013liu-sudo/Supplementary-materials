# Hydrogel cytokine prediction

Code and curated data for predicting macrophage TNF-alpha and IL-10 responses to hydrogels from ten material and experimental descriptors. The project compares four regression models and five imputation methods, with SHAP interpretation, t-SNE visualisation, case studies and Pareto optimisation.

| Target | Model | Numerical imputation |
| --- | --- | --- |
| TNF-alpha | GradientBoostingRegressor | Median |
| IL-10 | RandomForestRegressor | Median |

Model release: **2026-10-20**. The datasets contain 139 TNF-alpha and 133 IL-10 observations. Model settings are in `data/model_parameters.json`.

Python **3.13.9** · scikit-learn **1.7.2** · NumPy **2.3.5** · pandas **2.3.3** · SHAP **0.49.1**. Dependencies are pinned in `requirements.txt`; optional comparison models use XGBoost **3.4.1** and LightGBM **4.7.0**.

```bash
python -m pip install -r requirements.txt
python -m src.predict
python -m src.train --target both
```

Data are in `data/`, fitted pipelines in `models/`, and model and analysis code in `src/`.

## Hydrogel Cytokine Explorer

The Streamlit interface uses the saved 2026-10-20 pipelines and current public
category codes. It provides TNF-alpha and IL-10 predictions, the spread across
the three fitted models for each target, applicability-domain distances, and
grouped SHAP contributions. Inputs and results can be downloaded as CSV files.
Running the website does not train models.

```bash
python -m pip install -r requirements-app.txt
python -m streamlit run app.py
```

The website retains the three-column scientific illustration, Arial typography
and pale pink-purple theme. Navigation contains Home and Model Prediction;
descriptor definitions and the current supplementary document are available
from the prediction page. The approved transparent artwork is in `assets/`.
Its 3D icons are raster illustrations embedded alongside vector artwork and
editable SVG text; they represent descriptor concepts rather than measured
microstructures.

See [deployment instructions](docs/deployment.md) for Streamlit Community Cloud.
Model selection and evaluation records remain in
`data/validation_evidence/il10_20261020/release_provenance.json`.
