# Local-grid comparison and archived evidence

The optional comparison preserves the original scientific calculation: each of four model families is crossed with median, mean, iterative Bayesian ridge, scaled KNN3 and scaled KNN5 numerical imputation. Three local maximum-depth values are evaluated around each family's historical centre; all other parameters are model-specific. The complete candidate parameter dictionaries are in `data/local_grids.json`. No universal parameter set is applied across families.

Preprocessing is fitted inside each fold using only its Training rows. The categorical fill tokens, one-hot encoding and numeric column order are preserved. Scaled KNN uses Training-fitted StandardScaler, distance-weighted neighbours, inverse scaling and restoration of observed numeric values. Iterative imputation uses BayesianRidge, 20 maximum iterations, random state 42 and no posterior sampling. Every competitor receives the same ordered Training rows and fold assignments.

Each row's Validation score is pooled over all ten held-out fold predictions, not the mean of ten fold-specific R2 values. Ranking uses unrounded pooled R2 descending, then RMSE and MAE ascending. The command writes all candidates and the best candidate within each family/imputer pair for each fixed final seed. It does not use Test or external-case outcomes for grid ranking and does not update the accepted release.

`historical_local_grid.csv` contains the previously executed reference comparison for contexts A and B, including all 120 target/family/imputer/depth combinations. The archive's locked-representative flags identify historical Training-Validation choices. That pre-existing reference comparison preceded final-seed selection. To honour this repository's final-seed-only execution policy, the optional command evaluates the same candidate grid at the accepted seeds rather than reproducing the historical reference-seed run. Its numerical results therefore need not equal the historical grid table.

`final_seed_all20_comparisons.csv` contains the complete 20-row locked-pipeline Validation comparison at each of the six final seeds. It proves model and imputer optimality at the selected seeds. `data/locked_comparators.json` records each model/imputer pair's historically selected non-seed parameters. No other estimator-seed candidates or seed-screening code are distributed.

`data/validation_evidence/provenance.json` records the source filenames and hashes used to recover the executed comparison logic and evidence. `src/grid_search.py` is a readable refactoring, not an assertion that the original historical script had this file layout. Original controller and seed-extension scripts are deliberately excluded from the submission. The optional final-seed command was syntax-checked and its constructors and candidate dictionaries checked; it is not run by default.

## Controlled historical reproduction

All 120 original reference-grid configurations in contexts A and B were independently rerun at their historical reference seed, zero. Each used the unchanged ordered Training IDs and exact common folds. Two calculation paths were checked: the original cached fold preprocessing, and newly fitted fold preprocessing with the public pipeline constructors. Both paths produced identical OOF arrays to each other. Every pooled metric matched the original JSON checkpoint exactly, and all 40 family/imputer winner sets were identical. The grid logic, accepted parameters and final seeds were not changed.

The archived summary CSV has small decimal-export differences from those original JSON records: up to 5.55e-17 in R2, 5.68e-14 pg/mL in RMSE and 2.84e-14 pg/mL in MAE. These differences are retained in `controlled_grid_reproduction.csv`; they do not change any ranking. `grid_original_checkpoint_comparisons.csv` identifies the original checkpoints by relative path and SHA-256 and demonstrates zero metric differences. `controlled_grid_ranks.csv` and `controlled_grid_status.json` contain the complete ranking comparison and status.

The check used Python 3.13.9, NumPy 2.3.5, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.7.2, XGBoost 3.4.1 and LightGBM 4.7.0. No new seed range was screened, and no Test/case outcome was used in this reproduction check. The historical reference seed is audit provenance, not an additional final model seed.

These are best settings within the declared local grid under the shared Training-only CV procedure, not universal optima.
