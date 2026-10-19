# Hydrogel cytokine prediction

Code, data and fitted models accompanying *Supervised machine learning for predicting and explaining macrophage cytokine responses to hydrogels*. The project predicts TNF-alpha and IL-10 concentrations from ten descriptors of hydrogel materials and experimental conditions.

## Contributions

- Curated datasets of 139 TNF-alpha and 133 IL-10 observations, with source references and a shared descriptor codebook.
- Comparison of Gradient Boosting, XGBoost, Random Forest and LightGBM, plus separate Test results for the selected models.
- SHAP explanations, t-SNE landscapes and case studies of PEG microgel scaffolds and HPA hydrogels.
- A ten-descriptor Pareto analysis balancing lower predicted TNF-alpha against higher predicted IL-10 within the applicability domain.

## Model and data versions

| Component | Released version |
| --- | --- |
| Dataset | Thermal-harmonised v2 |
| Supplementary coding | `frequency_codes_20261019` |
| TNF-alpha | Gradient Boosting; median imputation; 1,000 trees; depth 3; learning rate 0.08 |
| IL-10 | Random Forest; iterative Bayesian ridge imputation; 250 trees; depth 9 |
| Case studies | C4: primary models; P15: separate whole-study-excluded models |
| Pareto | All ten descriptors; 16 profiles and three representatives |
| Runtime | Python 3.13.9; scikit-learn 1.7.2; NumPy 2.3.5; pandas 2.3.3; SHAP 0.49.1 |

Each primary model has three fitted realisations. Publication letters are decoded to the original model categories before prediction. The results describe retrospective model performance; known cross-target descriptor differences are recorded in the [data notes](docs/cross_target_descriptor_notes.md).

See the [supplementary document](docs/Supplementary_frequency_ordered_20261019.docx), [reproduction instructions](docs/reproducibility.md) and [pinned dependencies](requirements.txt).
