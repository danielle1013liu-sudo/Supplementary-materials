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
