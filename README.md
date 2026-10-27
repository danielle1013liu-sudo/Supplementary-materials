# Hydrogel cytokine models

## Background

This project models macrophage TNF-alpha and IL-10 concentrations from hydrogel properties and experimental conditions. The curated datasets contain 150 TNF-alpha observations and 131 IL-10 observations. Each observation has ten descriptors: hydrogel composition, storage modulus, pore size, bioactive cue, crosslinking method, crosslinking level, macrophage model, activation stimulus, exposure mode and exposure time.

The implementation includes Gradient Boosting, Random Forest, XGBoost and LightGBM regression. Cytokine concentrations are modelled in pg/mL without a target transformation.

## Selected models

| Target | Model | Numerical imputation |
| --- | --- | --- |
| TNF-alpha | Random Forest | Median |
| IL-10 | LightGBM | Iterative Bayesian ridge |

## Model parameters

| Target | Model | Main parameters |
| --- | --- | --- |
| TNF-alpha | Gradient Boosting | 500 estimators; learning rate 0.04; depth 4; minimum leaf size 2; subsample 0.9; feature fraction 0.8 |
| TNF-alpha | Random Forest | 400 estimators; depth 10; minimum leaf size 1; feature fraction 0.5; bootstrap disabled |
| TNF-alpha | XGBoost | 500 estimators; learning rate 0.04; depth 4; minimum child weight 1; row/column fractions 0.9; L2 penalty 3; histogram tree method |
| TNF-alpha | LightGBM | 500 estimators; learning rate 0.035; 15 leaves; minimum child size 3; row/column fractions 0.9; L2 penalty 2 |
| IL-10 | Gradient Boosting | 600 estimators; learning rate 0.06; depth 5; minimum leaf size 4; subsample 0.9; squared-error loss |
| IL-10 | Random Forest | 400 estimators; depth 10; minimum leaf size 1; feature fraction 0.6; bootstrap disabled |
| IL-10 | XGBoost | 1,200 estimators; learning rate 0.08; depth 4; minimum child weight 3; row/column fractions 0.6; L1 penalty 0.1; L2 penalty 5 |
| IL-10 | LightGBM | 800 estimators; learning rate 0.04; 15 leaves; minimum child size 3; row fraction 0.9; tree/node feature fractions 0.5; L2 penalty 1; maximum bins 15; minimum bin size 1 |

| Target | Partition random state | CV random state | Model random states |
| --- | --- | --- | --- |
| TNF-alpha | 3815 | 7 | 397, 106028, 154913 |
| IL-10 | 94641 | 99065 | 4247, 15433, 15645 |

The partition ratio is 80:20, with ten-fold cross-validation within Training. All four families use the same partitions and model random states for a given target. The three numerical descriptors use median, mean, iterative Bayesian ridge, or distance-weighted KNN imputation with three or five neighbours. Iterative imputation uses 20 iterations and random state 42. KNN distances are standardised within the fitting population. The seven categorical descriptors use a fixed one-hot schema, producing 35 model inputs in total.

Complete estimator arguments and family-specific imputation settings are recorded in [TNF-alpha configuration](config/tnf.json) and [IL-10 configuration](config/il10.json).

