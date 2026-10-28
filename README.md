# Hydrogel Cytokine Explorer

## Background

Hydrogels can shape macrophage responses through their network properties and incorporated bioactive cues. These responses also depend on macrophage activation and exposure conditions, making cytokine outcomes difficult to compare across formulations and studies. Quantitative assessment therefore requires both material descriptors and information about the biological assay. TNF-α and IL-10 provide complementary readouts of inflammatory output and regulatory signalling.

## Research aim

This project links hydrogel properties and experimental conditions to quantitative TNF-α and IL-10 responses using interpretable machine learning. The curated datasets contain 150 TNF-α observations and 131 IL-10 observations, described by ten material and experimental features. The aim is to support joint assessment of these cytokine responses and guide focused comparisons of immunomodulatory hydrogels.

## Main results

Random forest was selected for TNF-α prediction and LightGBM for IL-10 prediction.

| Target | Model | Test R² | RMSE (pg/mL) | MAE (pg/mL) |
| --- | --- | --- | --- | --- |
| TNF-α | Random forest | 0.9090 ± 0.0027 | 141.98 ± 2.13 | 98.38 ± 2.05 |
| IL-10 | LightGBM | 0.7749 ± 0.0181 | 208.85 ± 8.40 | 136.93 ± 1.52 |

Values are reported as mean ± SD.

Hydrogel Cytokine Explorer provides an interactive interface for predicting both cytokines, inspecting feature contributions and comparing outputs from user-specified formulations. The interface also reports training-data applicability-domain status and provides downloadable inputs, predictions and feature contributions. These functions support the prioritisation of hydrogel formulations for further experimental testing.

Design Screening retrieves non-dominated candidate profiles from the recorded candidate library under user-selected experimental and material constraints.

[Installation and reproduction](docs/reproduction.md).

